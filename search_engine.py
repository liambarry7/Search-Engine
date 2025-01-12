import engine_components as ec
import engines

def main():
    print("search engine main")
    ec.instal_nltk_datasets()

    while(True):
        print("\n--- Video Game Search Engine ---")
        engine_choice = input("Choose engine:"
                              "\n1 - Lemmatisation"
                              "\n2 - Stemming"
                              "\n3 - No lem/stem"
                              "\n4 - With stopwords"
                              "\n5 - With punctuation?"
                              "\n6 - No query expansion np"
                              "\n7 - No named entity recognition np"
                              "\n0 - Quit")

        if (engine_choice == '0'):
            break

        elif engine_choice == '1':
            engines.engine1()

        elif engine_choice == '2':
            print("2")
            engines.engine2()

        elif engine_choice == '3':
            print("3")
            engines.engine3()

        elif engine_choice == '4':
            print("4")
            engines.engine4()

        elif engine_choice == '5':
            print("5")
            engines.engine5()

        elif engine_choice == '6':
            print("6")

        elif engine_choice == '7':
            print("7")

        else:
            print("Please try again.")

main()