from googletrans import Translator
import asyncio
from langdetect import detect


def translation_needed(texts):

    if len(texts) > 10:
        texts = texts[:10]

    texts = " ".join(texts)

    if detect(texts) != "en":
        return True
    else:
        return False


async def translate_text(texts):
    async with Translator() as translator:
        result = await translator.translate(texts, dest="en")
    return result


# async def bulk_translate(text_lists):
#     translated_texts = await asyncio.gather(
#         *[translate_text(text) for text in text_lists]
#     )

#     return [translated_text.text for translated_text in translated_texts]


def bulk_translate(text_lists):
    translated_texts = []
    count = 0
    for i in text_lists:
        translated_text = asyncio.run(translate_text(i))
        translated_texts.append(translated_text.text)
        count += 1
        print(count)
    
    return translated_texts

if __name__ == "__main__":
    texts = ["नमस्कार वेलकम टू करियर 247 मैं हूं प्रशांत धवन कभी 2 बिलियन","डॉल के न्यूक्लियर बमबर्स 30 सेकंड में डिस्ट्रॉय होते हुए देखे हैं नहीं देखे मैं आपको दिखाता हूं इधर आप देख पाओगे यह फुटेज पूरी दुनिया में वायरल हो रही है।","नीचे जो यह प्लेन दिख रहे हैं ना यह न्यूक्लियर स्ट्रेटेजिक बमबर्स हैं। ऐसा कोई भी प्लेन भारत के पास नहीं है। गिनती के दो-तीन कंट्रीज के पास ऐसे बमबर्स हैं और यूक्रेन"]
    print(bulk_translate(texts))
