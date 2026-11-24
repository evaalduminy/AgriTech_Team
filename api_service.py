# ==============================================================================
# خدمة الـ Microservice لتشخيص أمراض أوراق البن باستخدام FastAPI
# مهيأة للربط المباشر مع تطبيق Flutter (حواء) ونظام Laravel (ردينة)
# ==============================================================================

# 1. استيراد المكتبات الأساسية
import os
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# استيراد فئة التنبؤ التي بنيناها في ملف predict_coffee_disease
from predict_coffee_disease import CoffeeDiseasePredictor, DEFAULT_MODEL_PATH


# ==============================================================================
# 2. تعريف نماذج البيانات (Pydantic Schemas) لتوثيق Swagger/OpenAPI
# ==============================================================================

class PredictionDetail(BaseModel):
    disease_name: str = Field(..., description="اسم المرض المشخص بالإنجليزية")
    confidence_score: float = Field(..., description="درجة التأكد كرقم عشري بين 0 و 1")
    confidence_percentage: str = Field(..., description="نسبة التأكد المئوية")
    clinical_note: Optional[str] = Field(None, description="ملاحظة أو تنبيه الحذر الزراعي")

class PredictionResponse(BaseModel):
    status: str = Field(..., example="success", description="حالة العملية (success أو rejected)")
    is_leaf: bool = Field(..., example=True, description="هل تم التحقق من أن الصورة ورقة نبات حقيقية")
    plant_pixel_ratio: float = Field(..., example=78.5, description="نسبة النسيج النباتي في الصورة %")
    message: Optional[str] = Field(None, description="رسالة توجيهية في حال تم رفض الصورة")
    prediction: Optional[PredictionDetail] = Field(None, description="بيانات المرض المتوقع")
    detailed_probabilities: Optional[Dict[str, float]] = Field(
        None, description="تفاصيل النسب المئوية لجميع الفئات الخمس"
    )

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    service: str = Field(..., example="Coffee Disease AI Microservice")
    version: str = Field(..., example="1.0.0")
    model_loaded: bool = Field(..., example=True)


# ==============================================================================
# 3. إدارة دورة حياة التطبيق (Lifespan Context Manager)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    تحميل النموذج مرة واحدة فقط في الذاكرة العشوائية عند إقلاع السيرفر.
    هذا يضمن استجابة فائقة السرعة للطلبات القادمة من تطبيق Flutter.
    """
    print("\n" + "=" * 60)
    print("[بدء التشغيل] جاري تهيئة خدمة الذكاء الاصطناعي وتحميل النموذج...")
    try:
        # تحميل النموذج وتخزينه في حالة التطبيق (App State)
        app.state.predictor = CoffeeDiseasePredictor(model_path=DEFAULT_MODEL_PATH)
        print("[جاهز] تم تحميل النموذج بنجاح وبدأت الخدمة في استقبال الطلبات.")
    except Exception as e:
        print(f"[تحذير] تعذر تحميل النموذج عند الإقلاع: {e}")
        print("[تنبيه] يرجى التأكد من تشغيل كود التدريب وتوليد ملف best_coffee_model.keras.")
        app.state.predictor = None
    print("=" * 60 + "\n")

    yield

    # كود الإغلاق والتنظيف عند إيقاف السيرفر
    print("\n[إيقاف التشغيل] جاري إغلاق خدمة الذكاء الاصطناعي...")


# ==============================================================================
# 4. تهيئة تطبيق FastAPI وإعدادات الأمان والتواصل (CORS)
# ==============================================================================

app = FastAPI(
    title="Coffee Leaf Disease Detection API",
    description="خدمة ذكاء اصطناعي (Microservice) مصممة لتصنيف أمراض أوراق البن اليمني وتغذية تطبيق Flutter ونظام Laravel.",
    version="1.0.0",
    lifespan=lifespan
)

# تفعيل الـ CORS للسماح بالاتصال من أي منصة (Flutter Web/Mobile أو Laravel Backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # السماح بجميع النطاقات أثناء التطوير
    allow_credentials=True,
    allow_methods=["*"],        # السماح بجميع دوال الـ HTTP (GET, POST, إلخ)
    allow_headers=["*"],
)


# ==============================================================================
# 5. نقاط النهاية للخدمة (API Endpoints)
# ==============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["مراقبة النظام (Health Check)"],
    summary="فحص صحة الخدمة وحالة النموذج"
)
async def health_check():
    """
    نقطة نهاية للتأكد من أن السيرفر يعمل وأن النموذج محمل وجاهز لاستقبال الصور.
    """
    is_ready = getattr(app.state, "predictor", None) is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "service": "Coffee Disease AI Microservice",
        "version": "1.0.0",
        "model_loaded": is_ready
    }


@app.post(
    "/api/v1/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["التشخيص والتنبؤ (Inference)"],
    summary="استقبال صورة ورقة البن وتشخيص المرض فوراً"
)
async def predict_coffee_leaf(
    file: UploadFile = File(..., description="ملف صورة ورقة البن المراد فحصها (JPEG أو PNG)")
):
    """
    تستقبل هذه النقطة الصورة المرفوعة من هاتف المزارع (عبر تطبيق Flutter)
    أو عبر نظام Laravel، وتقوم بتحليلها وإرجاع النتيجة خلال أجزاء من الثانية.
    """
    # 1. التحقق من وجود النموذج في الذاكرة
    predictor: CoffeeDiseasePredictor = getattr(app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="نموذج الذكاء الاصطناعي غير متوفر حالياً. يرجى التأكد من تدريب النموذج وتوفير ملف best_coffee_model.keras."
        )

    # 2. التحقق من نوع الملف المرسل (يجب أن يكون صورة)
    allowed_content_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"نوع الملف غير مدعوم ({file.content_type}). يرجى رفع ملف بصيغة JPG أو PNG."
        )

    try:
        # 3. قراءة محتوى الصورة كتدفق بايتات مباشرة في الذاكرة (In-Memory بدون تخزين على القرص)
        image_bytes = await file.read()

        if len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="ملف الصورة فارغ."
            )

        # 4. تنفيذ الاستدلال الرياضي واستخراج النتيجة
        prediction_result = predictor.predict_bytes(image_bytes)

        return prediction_result

    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"حدث خطأ غير متوقع أثناء معالجة الصورة: {str(error)}"
        )


# ==============================================================================
# 6. تشغيل السيرفر المباشر (Uvicorn Runner)
# ==============================================================================

if __name__ == "__main__":
    import uvicorn
    # تشغيل السيرفر المحلي على المنفذ 8000
    print("[تشغيل] جاري إطلاق السيرفر عبر Uvicorn...")
    uvicorn.run("api_service:app", host="0.0.0.0", port=8000, reload=True)
