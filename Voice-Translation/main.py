
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from gtts import gTTS 
from playsound import playsound
import speech_recognition as s_r
# This module is imported so that we can  
# play the converted audio 
import os 

# This language are supported
# Arabic (ar_AR), Czech (cs_CZ), German (de_DE), English (en_XX), Spanish (es_XX), Estonian (et_EE), Finnish (fi_FI),
# French (fr_XX), Gujarati (gu_IN), Hindi (hi_IN), Italian (it_IT), Japanese (ja_XX), Kazakh (kk_KZ), Korean (ko_KR), 
# Lithuanian (lt_LT), Latvian (lv_LV), Burmese (my_MM), Nepali (ne_NP), Dutch (nl_XX), Romanian (ro_RO), Russian (ru_RU), 
# Sinhala (si_LK), Turkish (tr_TR), Vietnamese (vi_VN), Chinese (zh_CN), Afrikaans (af_ZA), Azerbaijani (az_AZ), Bengali (bn_IN), 
# Persian (fa_IR), Hebrew (he_IL), Croatian (hr_HR), Indonesian (id_ID), Georgian (ka_GE), Khmer (km_KH), Macedonian (mk_MK), 
# Malayalam (ml_IN), Mongolian (mn_MN), Marathi (mr_IN), Polish (pl_PL), Pashto (ps_AF), Portuguese (pt_XX), Swedish (sv_SE), 
# Swahili (sw_KE), Tamil (ta_IN), Telugu (te_IN), Thai (th_TH), Tagalog (tl_XX), Ukrainian (uk_UA), Urdu (ur_PK), Xhosa (xh_ZA), 
# Galician (gl_ES), Slovene (sl_SI)

language_dict = {"english":"en_XX","spanish":"es_XX",
                 "french":"fr_XX","Portuguese":"pt_XX",
                 "hindi":"hi_IN","bengali":"bn_IN"}

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

# # translate Hindi to French
# tokenizer.src_lang = "hi_IN"
# encoded_hi = tokenizer(article_hi, return_tensors="pt")
# generated_tokens = model.generate(
#     **encoded_hi,
#     forced_bos_token_id=tokenizer.lang_code_to_id["fr_XX"]
# )
# output = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# print(output)

def convert_text_to_text(text,src_lang="en_XX",dst_lang="hi_IN"):
    tokenizer.src_lang = src_lang 
    encoded_hi = tokenizer(text,return_tensors='pt')
    generated_tokens = model.generate(
        **encoded_hi,
        forced_bos_token_id=tokenizer.lang_code_to_id[dst_lang]
    )
    output = tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
    return output

# # output = convert_text_to_text(article_hi,src_lang="hi_IN",dst_lang="en_XX")
# output = convert_text_to_text("who are you?",dst_lang="bn_IN")
# # print(output)
# myobj = gTTS(text=output[0], lang="bn", slow=False) 

# myobj.save("welcome.mp3") 
# playsound('welcome.mp3')

def input_text():
        r = s_r.Recognizer()
        said = " "
        with s_r.Microphone(device_index=1) as source:
            print("Say now!!!!")
            # r.adjust_for_ambient_noise(source)  # reduce noise
            audio = r.listen(source,phrase_time_limit=6)  # take voice input from the microphone
            try:
                said = r.recognize_google(audio,language='en-in')
            except Exception as e:
                print("Exception: " + str(e))
        return said

while True:
    input_src_lang = input("Input Source Language: ").lower()
    if input_src_lang in ["break","Break"]:
        break
    input_tar_lang = input("Input Target Language: ").lower()
    if input_tar_lang in ["break","Break"]:
        break

    if (input_src_lang in language_dict.keys()) and (input_tar_lang in language_dict.keys()):
        while True:

            src_lang = language_dict[input_src_lang]
            tar_lang = language_dict[input_tar_lang]  

            enter_the_text = input("Input You Source Text: ")
            # enter_the_text = input_text()
            # print(enter_the_text)

            if (enter_the_text != "break") and (enter_the_text != "Break"):
                output_text = convert_text_to_text(enter_the_text,src_lang=src_lang,dst_lang=tar_lang)
                my_obj = gTTS(output_text[0],lang=tar_lang[:2],slow=False)
                my_obj.save("translate_text.mp3")
                playsound("translate_text.mp3")
                os.remove("translate_text.mp3")
            else:
                break
    else:
        print("Check Your Language Choice section!")