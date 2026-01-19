from ..data.dataloader import PassageQuestionLoader

def main():
    loader = PassageQuestionLoader("speaker_listener_rl/data/passage_question_train.jsonl")

    for example in loader:
        passage = example["passage"]
        question = example["question"]
        print("PASSAGE:", passage)
        print("QUESTION:", question)
        break  #just tesitng first one
if __name__ == "__main__":
    main()
