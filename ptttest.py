 # -*- coding: utf-8 -*-
import logging
import scrapy
from ptt.items import PostItem
from datetime import datetime

class PTTSpider(scrapy.Spider):
    name = 'ptt'
    allowed_domains = ['ptt.cc']
    start_urls = ('https://www.ptt.cc/bbs/Stock/index1.html', )

    _pages = 0
    MAX_PAGES = 2970

    def parse(self, response):
            self._pages += 1
            for href in response.css('.r-ent > div.title > a::attr(href)'):
                url = response.urljoin(href.extract())
                yield scrapy.Request(url, callback=self.parse_post)

            if self._pages < PTTSpider.MAX_PAGES:
                next_page = response.xpath(
                    u'//div[@id="action-bar-container"]//a[contains(text(), "下頁")]/@href')
                if next_page:
                   url = response.urljoin(next_page[0].extract())
                   yield scrapy.Request(url, self.parse)
            else:
                logging.warning('max pages reached')

    def parse_post(self, response):
        item = PostItem()
        item['title'] = response.xpath(
            '//meta[@property="og:title"]/@content')[0].extract()
        item['author'] = response.xpath(
            u'//div[@class="article-metaline"]/span[text()="作者"]/following-sibling::span[1]/text()')[
                0].extract().split(' ')[0]
        datetime_str = response.xpath(
            u'//div[@class="article-metaline"]/span[text()="時間"]/following-sibling::span[1]/text()')[
                0].extract()
        item['date'] = datetime.strptime(datetime_str, '%a %b %d %H:%M:%S %Y')

        item['content'] = response.xpath('//div[@id="main-content"]/text()')[
            0].extract()

        comments = []
        total_score = 0
        for comment in response.xpath('//div[@class="push"]'):
            push_tag = comment.css('span.push-tag::text')[0].extract()
            push_user = comment.css('span.push-userid::text')[0].extract()
            push_content = comment.css('span.push-content::text')[0].extract()

            if u'推' in push_tag:
                score = 1
            elif u'噓' in push_tag:
                score = -1
            else:
                score = 0
            
            total_score += score

            comments.append({'user': push_user,
                             'content': push_content,
                             'score': score})
        item['comments'] = comments
        item['score'] = total_score
        item['url'] = response.url

        yield item
