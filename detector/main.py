import time
import os
import sys

def main():
    print("Sack Detection Edge Agent Starting...")
    
    # Check for picamera2
    try:
        import libcamera
        from picamera2 import Picamera2
        print("Picamera2 available. Running in PI mode.")
        has_camera = True
    except ImportError:
        print("Picamera2 NOT available. Running in DESKTOP mode.")
        has_camera = False

    # MQTT Setup
    mq_host = os.environ.get('MQTT_HOST', 'mqtt')
    mq_port = int(os.environ.get('MQTT_PORT', 1883))
    mq_user = os.environ.get('MQTT_USER')
    mq_pass = os.environ.get('MQTT_PASS')
    
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client()
        if mq_user and mq_pass:
            client.username_pw_set(mq_user, mq_pass)
            print(f"Configuring MQTT with user: {mq_user}")
            
        client.connect(mq_host, mq_port, 60)
        client.publish("sack/status", "online")
        client.loop_start()  # Start background thread for network loop
        print(f"Connected to MQTT Broker at {mq_host}:{mq_port}")
    except Exception as e:
        print(f"Failed to connect to MQTT: {e}")
        client = None

    while True:
        if has_camera:
            # Placeholder for real camera capture
            print("Capturing from PiCamera... (Stub)")
        else:
            # Placeholder for desktop/mock capture
            print("Capturing from DESKTOP Source... (Stub)")
        
        time.sleep(5)

if __name__ == "__main__":
    main()
