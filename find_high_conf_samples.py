import os
import glob
from train_gatekeeper_and_pipeline import predict_with_gatekeeper

all_test_fungal = glob.glob(r"D:\Coffee_Dataset_4Class\test\Fungal_Disease\*")
print(f"Total test fungal images: {len(all_test_fungal)}")

accepted_count = 0
for p in all_test_fungal[:20]:
    res = predict_with_gatekeeper(p)
    if res["status"] == "Accepted":
        accepted_count += 1
        print(f"✅ {os.path.basename(p)} -> Accepted: {res['decision']} ({res['confidence_percent']}), GK: {res['gatekeeper_score']:.3f}")

print(f"\nAccepted {accepted_count} out of 20 tested.")
