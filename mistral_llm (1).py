from mistralai import Mistral

API_KEY = "bKPW6dtz8uklLmPkUh8wFR6B6GDaQ6ky"
MODEL = "mistral-large-latest"

client = Mistral(api_key=API_KEY)

def query_mistral(prompt: str) -> str:
    try:
        response = client.chat.complete(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=600,
        )

        # Depuis la dernière mise à jour de l'API Mistral,
        # la structure correcte est :
        #
        # response.choices[0].message.content
        #
        return response.choices[0].message.content

    except Exception as e:
        print("ERREUR MISTRAL :", e)
        return ""
