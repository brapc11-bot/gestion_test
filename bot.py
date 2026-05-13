import discord
import requests

TOKEN = " MTUwMTIzMTIwODg1NDcyMDUzMg.Gjj6TS.x3yQJUzAxUXFTYSFp-S_S0wXXW6_LlC-vZIPYg"

API_URL = "http://127.0.0.1:8000/assistant/chat"

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

    query = message.content.strip()

    if not query:
        return

    try:
        await message.channel.send("Analyse en cours...")

        response = requests.post(
            API_URL,
            json={
                "user_id": str(message.author.id),
                "message": query
            },
            timeout=180
        )

        data = response.json()


        reply = data.get("response", "")

        if not reply:
            reply = "Erreur assistant: " + str(data)

        chunks = [reply[i:i+1900] for i in range(0, len(reply), 1900)]

        for chunk in chunks:
            await message.channel.send(chunk)

    except Exception as e:
        await message.channel.send(f"Erreur lors de l'analyse: {str(e)}")


client.run(TOKEN)
