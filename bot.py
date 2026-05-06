import discord
import requests

TOKEN = "MTUwMTIzMTIwODg1NDcyMDUzMg.Gjj6TS.x3yQJUzAxUXFTYSFp-S_S0wXXW6_LlC-vZIPYg"

API_URL = "http://127.0.0.1:8000/rag/assistant"

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


@client.event
async def on_ready():
    print(f"Bot connecte: {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    query = message.content

    try:
        response = requests.post(API_URL, json={
            "query": query,
            "n_results": 3
        })

        data = response.json()

        reply = "Analyse du probleme:\n\n"

        reply += "Types probables:\n"
        for t in data.get("type_probable", []):
            reply += f"- {t}\n"

        reply += "\nCauses:\n"
        for c in data.get("causes_probables", []):
            reply += f"- {c}\n"

        reply += "\nSolutions:\n"
        for s in data.get("solutions_recommandees", []):
            reply += f"- {s}\n"

        await message.channel.send(reply)

    except Exception as e:
        await message.channel.send("Erreur lors de l'analyse")


client.run(TOKEN)
