import os
from app import app as gr_blocks  # <-- your gr.Blocks

os.environ.setdefault("GRADIO_SERVER_NAME", "0.0.0.0")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

print("ATHENA ENV AT STARTUP:", {
    "ATHENA_S3_STAGING": os.getenv("ATHENA_S3_STAGING"),
    "ATHENA_REGION": os.getenv("ATHENA_REGION"),
    "ATHENA_WORKGROUP": os.getenv("ATHENA_WORKGROUP"),
    "ATHENA_DATABASE": os.getenv("ATHENA_DATABASE"),
})

# Start Gradio’s own HTTP server
gr_blocks.launch(
    server_name=os.environ["GRADIO_SERVER_NAME"],
    server_port=int(os.getenv("PORT", "8080")),
    prevent_thread_lock=True,
    quiet=True,
    share=False,
)

# Fallback handler (should rarely be hit once ready)
def handler(event, context):
    # Return 503 so we can tell if fallback is used
    return {"statusCode": 503, "body": "Service warming up"}
