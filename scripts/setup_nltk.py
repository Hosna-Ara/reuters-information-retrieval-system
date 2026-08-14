import nltk


def main():
    resources = ["reuters", "punkt_tab", "stopwords", "wordnet"]

    for resource in resources:
        print(f"Downloading NLTK resource: {resource}")
        nltk.download(resource)

    print("All NLTK resources downloaded successfully.")


if __name__ == "__main__":
    main()
