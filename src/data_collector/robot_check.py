import urllib.robotparser
import ssl
import urllib.request

class RobotCheck:
    def __init__(self, robots_txt_url):
        self.parser = urllib.robotparser.RobotFileParser()
        self.parser.set_url(robots_txt_url)

        # Create a context that doesn't verify SSL certificates
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

        # Use the custom context when reading the robots.txt file
        with urllib.request.urlopen(robots_txt_url, context=context) as response:
            self.parser.parse(response.read().decode('utf-8').splitlines())

    def is_allowed(self, url, user_agent):
        return self.parser.can_fetch(user_agent, url)

    def get_crawl_delay(self, user_agent):
        delay = self.parser.crawl_delay(user_agent)
        return delay if delay is not None else 1
