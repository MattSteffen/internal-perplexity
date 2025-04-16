import requests
from bs4 import BeautifulSoup
import json
from pathlib import Path
import time
from typing import List, Dict, Optional

class ConferenceTalkCrawler:
    def __init__(self, output_dir: str = "conference"):
        """
        Initialize the crawler with an output directory for saving talks.
        
        Args:
            output_dir (str): Directory where talk files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def get_soup(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch the webpage and return BeautifulSoup object.
        
        Args:
            url (str): URL of the talk page
            
        Returns:
            BeautifulSoup: Parsed HTML content
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return None

    def extract_talk_content(self, soup: BeautifulSoup) -> Dict:
        """
        Extract talk content from the BeautifulSoup object.
        
        Args:
            soup (BeautifulSoup): Parsed HTML content
            
        Returns:
            dict: Extracted talk content including title, author, role, and text
        """
        talk = {
            'title': '',
            'author': '',
            'author_role': '',
            'content': [],
            'url': ''
        }
        
        # Extract title
        title_elem = soup.find('h1', attrs={'data-aid': True})
        if title_elem:
            talk['title'] = title_elem.text.strip()
            
        # Extract author name
        author_elem = soup.find('p', class_='author-name')
        if author_elem:
            talk['author'] = author_elem.text.replace('By ', '').strip()
            
        # Extract author role
        role_elem = soup.find('p', class_='author-role')
        if role_elem:
            talk['author_role'] = role_elem.text.strip()
            
        # Extract talk paragraphs
        body_block = soup.find('div', class_='body-block')
        if body_block:
            paragraphs = body_block.find_all('p', attrs={'data-aid': True})
            talk['content'] = [p.text.strip() for p in paragraphs]
            
        return talk

    def save_talk(self, talk: Dict, url: str):
        """
        Save the talk content to a JSON file.
        
        Args:
            talk (dict): Talk content to save
            url (str): Original URL of the talk
        """
        talk['url'] = url
        filename = self.output_dir / f"{talk['title'].replace(' ', '_')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(talk, f, ensure_ascii=False, indent=2)
            
    def crawl_talks(self, urls: List[str], delay: float = 1.0):
        """
        Crawl multiple talk URLs and save their content.
        
        Args:
            urls (list): List of talk URLs to crawl
            delay (float): Delay between requests in seconds
        """
        for url in urls:
            print(f"Crawling: {url}")
            soup = self.get_soup(url)
            
            if soup:
                talk = self.extract_talk_content(soup)
                talk['url'] = url
                self.save_talk(talk, url)
                print(f"Saved: {talk['title']}")
            
            # Be polite and wait between requests
            time.sleep(delay)

# Example usage
if __name__ == "__main__":
    # Example URLs
    urls = [
        "https://www.churchofjesuschrist.org/study/general-conference/2024/10/47eyring",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/13holland?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/14johnson?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/15rasband?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/16cook?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/17gimenez?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/18eyring?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/21andersen?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/22lund?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/23palmer?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/24roman?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/25renlund?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/26boom?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/27uchtdorf?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/31stevenson?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/32wright?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/33rasband?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/34vargas?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/35christofferson?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/41bednar?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/42shumway?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/43runia?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/44causse?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/45gong?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/46mccune?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/47oaks?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/51soares?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/52strong?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/53whiting?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/54kim?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/55kearon?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/56tai?lang=eng",
        "https://www.churchofjesuschrist.org/study/general-conference/2025/04/57nelson?lang=eng",
    ]
    
    crawler = ConferenceTalkCrawler(output_dir="/Users/mattsteffen/projects/llm/internal-perplexity/data/conference")
    crawler.crawl_talks(urls)


"""
Output Schema:
{
  "title": "Simple Is the Doctrine of Jesus Christ",
  "author": "President Henry B. Eyring",
  "author_role": "Second Counselor in the First Presidency",
  "content": ["paragraph 1", "paragraph 2", ...],
  "url": "original_url_here"
}

"""