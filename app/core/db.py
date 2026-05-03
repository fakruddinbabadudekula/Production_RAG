from sqlalchemy.ext.asyncio import AsyncSession,create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import settings

# an asycn engine to connect the datbase.
a_engine=create_async_engine(
   url= settings.DATABASE_URL,
   echo=True
)

class Base(DeclarativeBase):
    pass

async def init_db():
    """Initialize the db. It creates the tables.Must be called before doing anything about the tables."""
    async with a_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        
    
# an async session maker, returns an contextmanager  
AsyncSessionMaker=sessionmaker(
    bind=a_engine,
    class_=AsyncSession,
     expire_on_commit=False
)
      
# dependency function return session, where we can interact with the database.
async def get_db():
    async with AsyncSessionMaker() as session:
        yield session 
        