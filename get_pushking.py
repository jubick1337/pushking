import html
import json
import re


def main():
    # Load the content of the file
    with open("./materials/pushking_original.txt", encoding="utf-8") as file:
        lines = file.readlines()

    poems_data = []

    # Iterate over the lines to group them into poems and find potential titles
    lines = [l for l in lines if l != '\n']
    i = 0
    while i < len(lines):
        poem = ''
        j = i
        if lines[i].startswith('\t\t'):
            while lines[j].startswith('\t\t'):
                poem += (
                    html.unescape(lines[j])
                    .replace('\t\t', '')
                    .replace('\xa0', ' ')
                    .replace('–', '-')
                    .replace('—', '-')
                    .replace('…', '...')
                    .replace('#', '')
                    .replace('"', '')
                    .replace('<', '')
                    .replace('>', '')
                    .replace('„', '')
                    .replace('3', 'З')
                )
                j += 1
                # Remove content inside square brackets
                poem = re.sub(r'\[.*?\]', '', poem)

            if i != j:
                k = i - 1
                names = []
                while not lines[k].startswith('\t\t') and k >= 0:
                    names.append(lines[k])
                    k -= 1
                poems_data.append(poem)
                i += j - i
        else:
            i += 1

    # Define a regular expression pattern for valid characters in the text
    valid_chars_pattern = re.compile(r"^['ёa-zA-ZúàâäéèêëîïôöùûüÿçœæА-Яа-я\s.,;?!()«»:-]+$")

    # Define a regular expression pattern for at least one Cyrillic character
    # cyrillic_pattern = re.compile(r'[а-яА-Я]')

    def is_valid_poem(poem):
        # Check if the poem contains at least one Cyrillic character

        # Check if the poem only contains valid characters
        match = valid_chars_pattern.match(poem)
        if not match:
            invalid_chars = re.findall(r"[^'ёa-zA-ZúàâäéèêëîïôöùûüÿçœæА-Яа-я\s.,;?!()«»:-]", poem)
            print(f"Invalid characters in poem: {invalid_chars}")
            return False

        return True

    with open("materials/pushking.json", "w", encoding="utf-8") as file:
        for data in poems_data:
            if len(data.split(' ')) > 8 and is_valid_poem(data):
                json.dump({'text': data}, file, ensure_ascii=False)
                file.write('\n')


if __name__ == "__main__":
    main()
