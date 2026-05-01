from search import search_prompt
from dotenv import load_dotenv
load_dotenv()

def main():
    input_question = input("Digite sua pergunta: ")
    if not input_question.strip():
        print("Pergunta vazia. Por favor, digite uma pergunta válida.")
        return

    #input_question = "Qual o faturamento da Empresa SuperTechIABrazil?"
    chain = search_prompt(input_question)
    
    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return
    
    answer = chain.invoke({"pergunta": input_question})

    print("PERGUNTA: "+ input_question)
    print("RESPOSTA: "+ answer.content)

if __name__ == "__main__":
    main()