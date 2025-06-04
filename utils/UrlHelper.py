import re


class UrlHelper:
    def __init__(self):
        pass

    def get_video_id(self, url):
        self.video_ids = re.findall(r"(?:v=|\/)([\w-]{11}).*", url)

        if self.video_ids:
            return self.video_ids[0]
        else:
            return None


if __name__ == "__main__":

    url_helper = UrlHelper()
    print(url_helper.get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
    print(url_helper.get_video_id("https://youtu.be/dQw4w9WgXcQ"))
    print(url_helper.get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=23"))
