import epitran
import nltk
import ssl
from nltk.corpus import cmudict
# Urdu phoneme conversion using Epitran
epi_urdu = epitran.Epitran('urd-Arab')  # Urdu phoneme converter
urdu_text = "یہ ایک جملہ ہے"
urdu_phonemes = epi_urdu.transliterate(urdu_text,ligatures=True)

# English phoneme conversion using CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

print(cmu_dict)

english_text = "This is a sentence".lower().split()
english_phonemes = [cmu_dict[word][0] for word in english_text if word in cmu_dict]

# Print phoneme transcriptions
print("Urdu Phonemes:", (urdu_phonemes))
print("English Phonemes:", english_phonemes)
