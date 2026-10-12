import asyncio
from neuronum import AgentIdentity

tools = [
    {
        "name": "jsonplaceholder_get_post",
        "description": "Look up a post by its ID.",
        "url": "https://jsonplaceholder.typicode.com/posts/{post_id}",
        "method": "GET",
        "parameters": {
            "type": "object",
            "properties": {
                "post_id": {
                    "type": "string",
                    "description": "The ID of the post to look up.",
                }
            },
            "required": ["post_id"],
        },
    }
]

async def main():
    async with AgentIdentity() as identity:
        session = await identity.create_secure_agent_session(
            guest="your@email.com", 
            subject="Testing Neuronum Agent Server", 
            instruct="You are a general purpose ai assistant",
            tools=tools,
        )
        print(session)

asyncio.run(main())