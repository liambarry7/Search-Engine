import engine_components as ec
import engines

def main():
    print("search engine main")
    ec.instal_nltk_datasets()

    while(True):
        print("\n--- Video Game Search Engine ---")
        engine_choice = input("Choose engine:"
                              "\n1 - Lemmatization"
                              "\n2 - Stemming"
                              "\n3 - No lem/stem"
                              "\n4 - With stopwords"
                              "\n5 - With punctuation"
                              "\n6 - With query expansion"
                              "\n7 - With metadata weighting"
                              "\n8 - No query expansion or metadata weighting"
                              "\n0 - Quit\n")

        if (engine_choice == '0'):
            break

        elif engine_choice == '1':
            engines.engine1()

        elif engine_choice == '2':
            engines.engine2()

        elif engine_choice == '3':
            engines.engine3()

        elif engine_choice == '4':
            engines.engine4()

        elif engine_choice == '5':
            engines.engine5()

        elif engine_choice == '6':
            engines.engine6()

        elif engine_choice == '7':
            engines.engine7()

        elif engine_choice == '8':
            engines.engine8()

        else:
            print("Please try again.")

main()