import urllib.robotparser


class RobotCheck:
    def __init__(self, robots_txt_url):
        self.parser = urllib.robotparser.RobotFileParser()
        self.parser.set_url(robots_txt_url)
        self.parser.read()

    def is_allowed(self, url, user_agent):
        return self.parser.can_fetch(user_agent, url)

    def get_crawl_delay(self, user_agent):
        delay = self.parser.crawl_delay(user_agent)
        return delay if delay is not None else 1
