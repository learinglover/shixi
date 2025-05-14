from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Header
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Base, User, History, DetectionRecord
import shutil
import os
from datetime import datetime
from ultralytics import YOLO
import cv2
import json
from typing import List, Optional

Base.metadata.create_all(bind=engine)
app = FastAPI()

# 加载模型（只需加载一次）
yolo_model = YOLO("../Models-----YOLO11/yolo11n.pt")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 用户注册
@app.post("/register")
def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="用户名已存在")
    user = User(username=username, password=password)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"success": True}

# 添加获取当前用户的函数
async def get_current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db)
) -> User:
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="未提供认证信息"
        )
    
    try:
        # 从请求头中获取用户ID
        user_id = int(authorization)
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=401,
                detail="用户不存在"
            )
        return user
    except ValueError:
        raise HTTPException(
            status_code=401,
            detail="无效的认证信息"
        )

# 修改登录接口，返回token
@app.post("/login")
def login(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username, User.password == password).first()
    if not user:
        raise HTTPException(status_code=400, detail="用户名或密码错误")
    return {"success": True, "user_id": user.id, "token": str(user.id)}  # 使用用户ID作为token

# 修改密码
@app.put("/users/{user_id}/password")
def change_password(user_id: int, new_password: str = Form(...), db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    user.password = new_password
    db.commit()
    return {"success": True}

# 注销
@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    db.delete(user)
    db.commit()
    return {"success": True}

def is_image(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))

def is_video(filename):
    return filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))

@app.post("/upload")
def upload_file(
    user_id: int = Form(...),
    file: UploadFile = File(...),
    conf: float = Form(0.25),      # 置信度阈值
    iou: float = Form(0.45),       # IoU阈值
    max_det: int = Form(300),      # 最大检测数
    db: Session = Depends(get_db)
):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    input_path = os.path.join(upload_dir, f"input_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pedestrian_count = 0
    vehicle_count = 0
    output_path = input_path.replace("input_", "output_")

    if is_image(file.filename):
        results = yolo_model(
            input_path,
            conf=conf,
            iou=iou,
            max_det=max_det
        )
        for r in results:
            for c in r.boxes.cls:
                label = yolo_model.model.names[int(c)]
                if label == 'person':
                    pedestrian_count += 1
                elif label in ['car', 'bus', 'truck', 'motorcycle']:
                    vehicle_count += 1
        annotated_img = results[0].plot()
        cv2.imwrite(output_path, annotated_img)
    elif is_video(file.filename):
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            results = yolo_model(
                frame,
                conf=conf,
                iou=iou,
                max_det=max_det
            )
            for r in results:
                for c in r.boxes.cls:
                    label = yolo_model.model.names[int(c)]
                    if label == 'person':
                        pedestrian_count += 1
                    elif label in ['car', 'bus', 'truck', 'motorcycle']:
                        vehicle_count += 1
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
        cap.release()
        out.release()
    else:
        raise HTTPException(status_code=400, detail="仅支持图片或视频文件")

    history = History(
        user_id=user_id,
        input_file=input_path,
        output_file=output_path,
        pedestrian_count=pedestrian_count,
        vehicle_count=vehicle_count,
        confidence_threshold=conf,
        iou_threshold=iou,
        max_detections=max_det
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return {
        "history_id": history.id,
        "output_file": output_path,
        "pedestrian_count": pedestrian_count,
        "vehicle_count": vehicle_count,
        "confidence_threshold": conf,
        "iou_threshold": iou,
        "max_detections": max_det
    }

# 获取历史记录
@app.get("/history/{user_id}")
def get_history(user_id: int, db: Session = Depends(get_db)):
    histories = db.query(History).filter(History.user_id == user_id, History.is_deleted == False).all()
    return histories

# 回收记录（移入回收站）
@app.put("/history/{history_id}/remove")
def remove_history(history_id: int, db: Session = Depends(get_db)):
    history = db.get(History, history_id)
    if not history:
        raise HTTPException(status_code=404, detail="记录不存在")
    history.is_deleted = True
    db.commit()
    return {"success": True}

# 查看回收站
@app.get("/recycle/{user_id}")
def get_recycle(user_id: int, db: Session = Depends(get_db)):
    histories = db.query(History).filter(History.user_id == user_id, History.is_deleted == True).all()
    return histories

# 恢复记录
@app.put("/recycle/{history_id}/restore")
def restore_history(history_id: int, db: Session = Depends(get_db)):
    history = db.get(History, history_id)
    if not history:
        raise HTTPException(status_code=404, detail="记录不存在")
    history.is_deleted = False
    db.commit()
    return {"success": True}

# 永久删除记录
@app.delete("/recycle/{history_id}")
def delete_history(history_id: int, db: Session = Depends(get_db)):
    history = db.get(History, history_id)
    if not history:
        raise HTTPException(status_code=404, detail="记录不存在")
    db.delete(history)
    db.commit()
    return {"success": True}

@app.post("/api/detect")
async def detect_objects(
    files: List[UploadFile] = File(...),
    confidence: float = 0.25,
    iou_threshold: float = 0.45,
    max_detections: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        results = []
        for file in files:
            # 保存上传的文件
            file_location = f"uploads/{file.filename}"
            with open(file_location, "wb+") as file_object:
                file_object.write(await file.read())
            
            # 处理检测结果
            result = {
                "filename": file.filename,
                "detections": [],
                "pedestrian_count": 0,
                "vehicle_count": 0
            }
            
            if is_image(file.filename):
                # 图片检测
                detections = yolo_model(
                    file_location,
                    conf=confidence,
                    iou=iou_threshold,
                    max_det=max_detections
                )
                
                for det in detections:
                    if len(det.boxes) > 0:
                        for box in det.boxes:
                            if box.conf[0] >= confidence:
                                label = yolo_model.model.names[int(box.cls[0])]
                                # 统计行人和车辆数量
                                if label == 'person':
                                    result["pedestrian_count"] += 1
                                elif label in ['car', 'bus', 'truck', 'motorcycle']:
                                    result["vehicle_count"] += 1
                                    
                                result["detections"].append({
                                    "class": label,
                                    "confidence": float(box.conf[0]),
                                    "bbox": box.xyxy[0].tolist()
                                })
                
                # 保存检测后的图片
                output_path = file_location.replace("input_", "output_")
                annotated_img = detections[0].plot()
                cv2.imwrite(output_path, annotated_img)
                
            elif is_video(file.filename):
                # 视频检测
                cap = cv2.VideoCapture(file_location)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                output_path = file_location.replace("input_", "output_")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_count = 0
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                        
                    # 每隔几帧处理一次，提高效率
                    if frame_count % 3 == 0:  # 每3帧处理一次
                        detections = yolo_model(
                            frame,
                            conf=confidence,
                            iou=iou_threshold,
                            max_det=max_detections
                        )
                        
                        for det in detections:
                            if len(det.boxes) > 0:
                                for box in det.boxes:
                                    if box.conf[0] >= confidence:
                                        label = yolo_model.model.names[int(box.cls[0])]
                                        # 统计行人和车辆数量
                                        if label == 'person':
                                            result["pedestrian_count"] += 1
                                        elif label in ['car', 'bus', 'truck', 'motorcycle']:
                                            result["vehicle_count"] += 1
                                            
                                        result["detections"].append({
                                            "class": label,
                                            "confidence": float(box.conf[0]),
                                            "bbox": box.xyxy[0].tolist()
                                        })
                        
                        annotated_frame = detections[0].plot()
                        out.write(annotated_frame)
                    else:
                        out.write(frame)
                        
                    frame_count += 1
                
                cap.release()
                out.release()
            else:
                raise HTTPException(status_code=400, detail="不支持的文件类型")
            
            results.append(result)
            
            # 保存检测记录
            history = History(
                user_id=current_user.id,
                input_file=file_location,
                output_file=output_path,
                pedestrian_count=result["pedestrian_count"],
                vehicle_count=result["vehicle_count"],
                confidence_threshold=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections
            )
            db.add(history)
            db.commit()
            
            # 删除临时文件
            os.remove(file_location)
            
        return {
            "status": "success", 
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
