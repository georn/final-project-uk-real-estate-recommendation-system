import time
import requests
from bs4 import BeautifulSoup
from robot_check import RobotCheck
from ratelimit import limits, sleep_and_retry

# 10 requests per minute
REQUESTS_PER_MINUTE = 10


@sleep_and_retry
@limits(calls=REQUESTS_PER_MINUTE, period=60)
def make_request(url, headers):
    return requests.get(url, headers=headers)


def get_property_urls(base_url, search_url, user_agent):
    headers = {'User-Agent': user_agent}
    robot_check = RobotCheck("https://www.onthemarket.com/robots.txt")
    property_urls = set()
    page_number = 1

    while search_url:
        if robot_check.is_allowed(search_url, user_agent):
            response = make_request(search_url, headers=headers)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                links = soup.select('div.otm-PropertyCardMedia > div > a')
                current_page_urls = set()

                for link in links:
                    href = link.get('href')
                    if href and href.startswith('/details/'):  # Filtering other links
                        full_url = 'https://www.onthemarket.com' + href
                        property_urls.add(full_url)
                        current_page_urls.add(full_url)

                # print(f"{len(current_page_urls)} URLs captured on Page {page_number}.")

                # Find the next page URL
                next_page = soup.select_one('a[title="Next page"]')
                if next_page:
                    search_url = base_url + next_page['href']
                    page_number += 1
                else:
                    search_url = None

            else:
                print(f"Failed to retrieve the webpage: Status Code {response.status_code}")
                break

            # Sleep for the crawl delay
            time.sleep(robot_check.get_crawl_delay(user_agent))
        else:
            print("Scraping is disallowed by the website's robots.txt for this URL.")
            break

    return list(property_urls)
