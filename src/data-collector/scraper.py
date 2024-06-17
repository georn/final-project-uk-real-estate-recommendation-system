import requests
from bs4 import BeautifulSoup


def scrape_property_details(property_url, headers, counter):
    try:
        response = requests.get(property_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extracting data
            # Extracting the title
            title = soup.find('h1', class_='h4 md:text-xl leading-normal').get_text(strip=True) if soup.find('h1', class_='h4 md:text-xl leading-normal') else 'Title not available'

            # Extracting the address
            address_div = soup.find('div', class_='text-slate h4 font-normal leading-none font-heading')
            address = address_div.get_text(strip=True) if address_div else 'Address not available'

            # Extracting the price
            price_div = soup.find('div', class_='otm-Price')
            if price_div:
                price = price_div.find('span', class_='price').get_text(strip=True) if price_div.find('span', class_='price') else 'Price not available'

                # Extracting the pricing qualifier if present
                qualifier_div = price_div.find('small', class_='qualifier')
                price_qualifier = qualifier_div.get_text(strip=True) if qualifier_div else 'Price qualifier not available'
            else:
                price = 'Price not available'
                price_qualifier = 'Price qualifier not available'

            # Extracting the listing time
            listing_time_div = soup.find('div', class_='text-denim')
            if listing_time_div:
                listing_time = listing_time_div.find('small' , class_='font-heading').get_text(strip=True) if listing_time_div.find(
                    'small') else 'Listing time not available'
            else:
                listing_time = 'Listing time not available'

            # Extracting the property type
            property_type_div = soup.find('div', class_='otm-PropertyIcon')
            property_type = property_type_div.get_text(
                strip=True) if property_type_div else 'Property type not available'

            details_div = soup.find('div', class_='font-heading text-xs flex flex-wrap border-t border-b mb-6 md:text-md py-3 md:py-4 md:mb-9')
            if details_div:
                # Extracting the number of bedrooms
                bedrooms = 'Bedrooms info not available'
                for div in details_div.find_all('div'):
                    if 'bed' in div.get_text(strip=True).lower():
                        bedrooms = div.get_text(strip=True)
                        break

                # Extracting the number of bathrooms
                bathrooms = 'Bathrooms info not available'
                for div in details_div.find_all('div'):
                    if 'bath' in div.get_text(strip=True).lower():
                        bathrooms = div.get_text(strip=True)
                        break

                # Extracting the EPC rating
                epc_rating = 'EPC rating not available'
                for div in details_div.find_all('div'):
                    if 'epc rating' in div.get_text(strip=True).lower():
                        epc_rating = div.get_text(strip=True)
                        break

                # Extracting the property size
                size = 'Size info not available'
                for div in details_div.find_all('div'):
                    text = div.get_text(strip=True).lower()
                    if 'sq ft' in text or 'sq m' in text:
                        size = div.get_text(strip=True)
                        break

            # Extracting features
            features_section = soup.find('section', class_='otm-FeaturesList')
            features = []
            if features_section:
                feature_items = features_section.find_all('li', class_='otm-ListItemOtmBullet')
                for item in feature_items:
                    feature_text = item.get_text(strip=True)
                    features.append(feature_text)

            return {
                'id': counter,
                'property_url': property_url,
                'title': title,
                'address': address,
                'price': price,
                'pricing_qualifier': price_qualifier,
                'listing_time': listing_time,
                'property_type': property_type,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'epc_rating': epc_rating,
                'size': size,
                'features': features
            }
        else:
            print(f"Failed to retrieve the property page: Status Code {response.status_code}")
            return {}
    except requests.exceptions.RequestException as e:
        return {'error': f"Request failed: {e}"}
    except Exception as e:
        return {'error': f"An unexpected error occurred: {e}"}
