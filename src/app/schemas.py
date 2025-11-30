from pydantic import BaseModel

class FruitFeatures(BaseModel):
    size_cm: float
    weight_g: float
    avg_price: float
    shape: str
    color: str
    taste: str
