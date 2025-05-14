from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    password = Column(String(50))  
    histories = relationship("History", back_populates="owner")

class History(Base):
    __tablename__ = "histories"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    input_file = Column(String(200))
    output_file = Column(String(200))
    pedestrian_count = Column(Integer, default=0)
    vehicle_count = Column(Integer, default=0)
    confidence_threshold = Column(Float, default=0.25)
    iou_threshold = Column(Float, default=0.45)
    max_detections = Column(Integer, default=100)
    created_at = Column(DateTime, default=datetime.now)
    is_deleted = Column(Boolean, default=False)
    owner = relationship("User", back_populates="histories")

class DetectionRecord(Base):
    __tablename__ = "detection_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    image_path = Column(String(200))
    confidence_threshold = Column(Float, default=0.25)
    iou_threshold = Column(Float, default=0.45)
    max_detections = Column(Integer, default=100)
    detection_results = Column(Text)  # 存储JSON格式的检测结果
    created_at = Column(DateTime, default=datetime.now)
    
    user = relationship("User")
