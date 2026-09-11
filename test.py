import asyncio
from neuronum import AgentIdentity

async def main():
    async with AgentIdentity() as identity:
        session = await identity.create_secure_agent_session(
            guest="your@email.com", 
            subject="Testing Neuronum Agent Server", 
            instruct="You are a general purpose ai assistant"
        )
        print(session)

asyncio.run(main())