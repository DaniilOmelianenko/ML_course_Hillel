from transformers import (pipeline, Pipeline, AutoConfig, AutoTokenizer, AutoProcessor, AutoModelForSeq2SeqLM,
                          TranslationPipeline)

t5_large_model: str = "t5-large"
T5_SMALL_MODEL: str = "t5-small"
device: str = "cpu"


class Translator:

    def __init__(
            self,
            model=T5_SMALL_MODEL,
            task="translation_en_to_de",
            tokenizer=AutoTokenizer,
            model_config=AutoConfig,
            processor=AutoProcessor,
    ):
        model_config = self.configurate_recognizer_part(recognizer_part_class=AutoConfig, model=model)
        tokenizer = self.configurate_recognizer_part(recognizer_part_class=AutoTokenizer, model=model)
        processor = self.configurate_recognizer_part(recognizer_part_class=AutoProcessor, model=model)
        model = self.configurate_recognizer_part(recognizer_part_class=AutoModelForSeq2SeqLM, model=model)

        self.translation_pipeline: Pipeline = pipeline(
            task=task,
            config=model_config,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

    @staticmethod
    def configurate_recognizer_part(recognizer_part_class, model):
        local_model_path: str = f"./models/{model}"

        try:
            return recognizer_part_class.from_pretrained(pretrained_model_name_or_path=local_model_path)

        except Exception as exception:
            print(type(exception), exception)
            part = recognizer_part_class.from_pretrained(model)
            part.save_pretrained(save_directory=local_model_path)
            return recognizer_part_class.from_pretrained(pretrained_model_name_or_path=local_model_path)

    def translate(self, text: str):
        print(self.translation_pipeline(text))

    def translate_sequence(self, text: list):
        for sentence in text:
            print(self.translation_pipeline(sentence))


# translator: Translator = Translator()
# # print(translator.translate(text="Hello"))
# translator.translate_sequence("Hello! Good Morning! My name is Daniil Omelianenko! I'm Ukrainian!".split(sep="!"))
# translator.translate("Hello! Good Morning! My name is Daniil Omelianenko! I'm Ukrainian!")

anthem_ukr: str = """
Ще не вмерла України і слава, і воля.
Ще нам, браття молодії, усміхнеться доля.
Згинуть наші вороженьки, як роса на сонці,
Запануєм і ми, браття, у своїй сторонці.

Приспів:
Душу й тіло ми положим за нашу свободу,
І покажем, що ми, браття, козацького роду.

Станем, браття, в бій кривавий від Сяну до Дону,
В ріднім краю панувати не дамо нікому;
Чорне море ще всміхнеться, дід Дніпро зрадіє,
Ще у нашій Україні доленька наспіє.

Приспів.
А завзяття, праця щира свого ще докаже,
Ще ся волі в Україні піснь гучна розляже,
За Карпати відоб’ється, згомонить степами,
України слава стане поміж ворогами.

Приспів.
"""
ENG: str = "en"
UKR: str = "uk"

pipe: TranslationPipeline = pipeline(task="translation", model="facebook/nllb-200-distilled-600M")
print(pipe(anthem_ukr, src_lang=ENG, tgt_lang=UKR, max_length=400))
