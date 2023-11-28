import eng_to_ipa as ipa
import sys

def text_to_ipa(text):
    ipa_list = list(ipa.convert(text))
    for i in range(len(ipa_list)):
        if ipa_list[i] == ' ':
            ipa_list[i] = '_'

    # Check if certain characters occur next to each other and combine them: aʊ, ɔɪ, aɪ, tʃ, dʒ
    for i in range(len(ipa_list)-1):
        if ipa_list[i] == 'a' and ipa_list[i+1] == 'ʊ':
            ipa_list[i] = 'aʊ'
            ipa_list[i+1] = ''
        elif ipa_list[i] == 'ɔ' and ipa_list[i+1] == 'ɪ':
            ipa_list[i] = 'ɔɪ'
            ipa_list[i+1] = ''
        elif ipa_list[i] == 'a' and ipa_list[i+1] == 'ɪ':
            ipa_list[i] = 'aɪ'
            ipa_list[i+1] = ''
        elif ipa_list[i] == 't' and ipa_list[i+1] == 'ʃ':
            ipa_list[i] = 'tʃ'
            ipa_list[i+1] = ''
        elif ipa_list[i] == 'd' and ipa_list[i+1] == 'ʒ':
            ipa_list[i] = 'dʒ'
            ipa_list[i+1] = ''

    # Remove empty strings
    ipa_list = list(filter(None, ipa_list))

    return ipa_list

def open_ipa_viseme_file():
    with open("ipa_viseme.csv", "r") as f:
        mappings = [i.split(',') for i in f.read().splitlines()]

    d = {}
    for i in range(len(mappings)):
        for j in range(1,len(mappings[i])):  
            d[mappings[i][j]] = i

    return d

def ipa_list_to_visemes(ipa_list, mappings):
    viseme_list = []
    for i in ipa_list:
        if i in mappings:
            viseme_list.append(mappings[i])
        # else:
        #     print("Error: IPA character not found in mappings: " + i)

    return viseme_list

def convert_text_to_visemes(text):
    ipa_str = text_to_ipa(text)
    ipa_viseme = open_ipa_viseme_file()
    viseme_list = ipa_list_to_visemes(ipa_str, ipa_viseme)

    return viseme_list
    

def main():
    text = sys.argv[1]
    viseme_list = convert_text_to_visemes(text)
    print(viseme_list)

if __name__ == "__main__":
    main()