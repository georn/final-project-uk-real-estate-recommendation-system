import json
from crawler import get_property_urls
from scraper import scrape_property_details
from robot_check import RobotCheck
from tqdm import tqdm
import time
import sys
import os

class DataCollectorService:
    def __init__(self, base_url, user_agent, max_price=120000):
        self.base_url = base_url
        self.user_agent = user_agent
        self.max_price = max_price
        self.robot_check = RobotCheck(f"{base_url}/robots.txt")

    @staticmethod
    def generate_price_segments(max_price, segment_size=100000):
        segments = []
        current_min = 0
        while current_min < max_price:
            current_max = min(current_min + segment_size, max_price)
            segments.append((current_min, current_max))
            current_min = current_max + 1
        return segments

    def collect_data_segment(self, min_price, max_price):
        search_url = f"{self.base_url}/for-sale/property/buckinghamshire/?auction=false&min-price={min_price}&max-price={max_price}&new-home-flag=F&prop-types=bungalows&prop-types=detached&prop-types=flats-apartments&prop-types=semi-detached&prop-types=terraced&retirement=false&shared-ownership=false"
        return get_property_urls(self.base_url, search_url, self.user_agent)

    def collect_data(self):
        price_segments = self.generate_price_segments(self.max_price)
        all_property_urls = set()
        all_property_data = []
        counter = 1

        for min_price, max_price in price_segments:
            segment_urls = self.collect_data_segment(min_price, max_price)
            all_property_urls.update(segment_urls)
            print(f"Segment {min_price}-{max_price}: Found {len(segment_urls)} URLs.")

        print(f"Found a total of {len(all_property_urls)} property URLs from all segments.")

        for url in tqdm(all_property_urls, desc="Scraping properties"):
            if self.robot_check.is_allowed(url, self.user_agent):
                headers = {'User-Agent': self.user_agent}
                data = scrape_property_details(url, headers, counter)
                if 'error' in data:
                    print(data['error'])
                else:
                    all_property_data.append(data)
                    counter += 1
                    # Respect the crawl delay
                    time.sleep(self.robot_check.get_crawl_delay(self.user_agent))
            else:
                tqdm.write(f"Skipping {url}, disallowed by robots.txt.")

        self.save_data(all_property_data)

    def save_data(self, data):
        data_directory = './data'
        filename = f'./data/property_data_{self.max_price}.json'
        backup_filename = f'./data/property_data_{self.max_price}-backup.json'

        try:
            # Ensure data directory exists
            if not os.path.exists(data_directory):
                os.makedirs(data_directory)
                print(f"Created directory {data_directory}")

            # Rename existing files if they exist
            if os.path.exists(backup_filename):
                os.remove(backup_filename)
            if os.path.exists(filename):
                os.rename(filename, backup_filename)
                print(f"Renamed existing file to {backup_filename}")

            # Save the new data
            with open(filename, 'w') as file:
                json.dump(data, file, indent=4)
            print(f"Data saved successfully in {filename}")
        except IOError as e:
            print(f"An IOError occurred while saving data: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while saving data: {e}")

if __name__ == "__main__":
    max_price_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 120000
    base_url = 'https://www.onthemarket.com'
    search_url = base_url + '/for-sale/property/buckinghamshire/?auction=false&max-price=500000&min-price=10000&new-home-flag=F&prop-types=bungalows&prop-types=detached&prop-types=flats-apartments&prop-types=semi-detached&prop-types=terraced&retirement=false&shared-ownership=false'
    user_agent = 'StudentDataProjectScraper/1.0 (Contact: gor5@student.london.ac.uk)'
    service = DataCollectorService(base_url, user_agent, max_price_arg)
    service.collect_data()
