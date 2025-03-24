# Flask-based payment system with profiles, names, blockchain feature, notifications, transaction limit, and profile editing

from flask import Flask, render_template, request, send_file, session, jsonify
import random
import os
import json
import csv
from io import StringIO
from datetime import datetime
import hashlib
import cv2
import numpy as np
import base64
from io import BytesIO
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from pymongo import MongoClient
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure key in production

# MongoDB connection
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
MONGO_URI = "mongodb+srv://aidpaymentuser:<cUdkMfZvVWBjwW6t>@cluster0.yyyra.mongodb.net/?retryWrites=true&w=majority&appName=cluster0"
client = MongoClient(MONGO_URI)
db = client['AidPaymentSystem']
users_collection = db['users']
transactions_collection = db['transactions']
blockchain_collection = db['blockchain']
offline_transactions_collection = db['offline_transactions']

# Global variables
users = {}
ledger = []
offline_queue = []
output_messages = []
transactions = []
is_face_recognizer_trained = False

# Twilio configuration
TWILIO_ACCOUNT_SID = "AC733d521c3f4d3f6e344119bce45fa04c"
TWILIO_AUTH_TOKEN = "33e6f86ccb9d8156a5552f5d76260ab7"
TWILIO_PHONE_NUMBER = "+19898122743"

# Email configuration
EMAIL_ADDRESS = "nalluriprudhvichanakya@gmail.com"
EMAIL_PASSWORD = "vyjy wlkn lmtu ggun"

# Initialize Twilio client
try:
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
except Exception as e:
    print(f"Error initializing Twilio client: {e}")
    twilio_client = None

# Function to send SMS
def send_sms(to_phone, message):
    if twilio_client is None:
        print("Twilio client not initialized. SMS not sent.")
        return
    try:
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone
        )
        print(f"SMS sent to {to_phone}: {message}")
    except Exception as e:
        print(f"Error sending SMS to {to_phone}: {e}")

# Function to send email
def send_email(to_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, to_email, msg.as_string())
        print(f"Email sent to {to_email}: {subject}")
    except Exception as e:
        print(f"Error sending email to {to_email}: {e}")

# Initialize face recognizer and face detector
try:
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Failed to load Haar cascade file.")
    print("Face recognizer and cascade loaded successfully.")
except AttributeError as e:
    print(f"Error: OpenCV face module not available. Please install opencv-contrib-python: {e}")
    face_recognizer = None
    face_cascade = None
except Exception as e:
    print(f"Error initializing face recognizer: {e}")
    face_recognizer = None
    face_cascade = None

# Function to compute SHA-256 hash
def compute_hash(block):
    block_string = json.dumps(block, sort_keys=True)
    return hashlib.sha256(block_string.encode()).hexdigest()

# Function to save face images
def save_face_images(user_id, face_images):
    if face_recognizer is None or face_cascade is None:
        print("Face recognizer or cascade not available. Cannot save face images.")
        return 0

    user_dir = os.path.join("faces", str(user_id))
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    valid_faces = 0
    for i, image_data in enumerate(face_images):
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            print(f"Failed to decode image {i+1} for user ID {user_id}: {str(e)}")
            continue

        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to decode image {i+1} into OpenCV format for user ID {user_id}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=1, minSize=(15, 15))
        if len(faces) == 0:
            print(f"No face detected in image {i+1} for user ID {user_id}")
            continue

        image_path = os.path.join(user_dir, f"face_{i+1}.jpg")
        success = cv2.imwrite(image_path, image)
        if success:
            valid_faces += 1
        else:
            print(f"Failed to save face image: {image_path}")

    return valid_faces

# Function to train the face recognizer
def train_face_recognizer():
    global is_face_recognizer_trained
    if face_recognizer is None or face_cascade is None:
        print("Face recognizer not available. Skipping training.")
        is_face_recognizer_trained = False
        return

    faces = []
    labels = []
    label_map = {}
    label_index = 0

    faces_dir = "faces"
    if not os.path.exists(faces_dir):
        os.makedirs(faces_dir)

    for user_id, user in users.items():
        user_dir = os.path.join(faces_dir, str(user_id))
        if not os.path.exists(user_dir):
            continue

        label_map[label_index] = user_id
        for filename in os.listdir(user_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(user_dir, filename)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                image = cv2.equalizeHist(image)
                detected_faces = face_cascade.detectMultiScale(image, scaleFactor=1.03, minNeighbors=1, minSize=(15, 15))
                if len(detected_faces) == 0:
                    continue

                (x, y, w, h) = detected_faces[0]
                face_roi = image[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100, 100))
                faces.append(face_roi)
                labels.append(label_index)

        label_index += 1

    if faces and labels:
        face_recognizer.train(faces, np.array(labels))
        with open("label_map.json", "w") as f:
            json.dump(label_map, f)
        is_face_recognizer_trained = True
    else:
        is_face_recognizer_trained = False

# Function to migrate existing data to MongoDB
def migrate_data_to_mongodb():
    # Migrate users
    if os.path.exists("users.json"):
        with open("users.json", "r") as file:
            users_data = json.load(file)
            users_data = {int(k): v for k, v in users_data.items()}
        users_collection.delete_many({})  # Clear existing data
        for user_id, user in users_data.items():
            user["_id"] = user_id  # Use user_id as the MongoDB document _id
            users_collection.update_one({"_id": user_id}, {"$set": user}, upsert=True)
        print("Migrated users to MongoDB")

    # Migrate transactions
    if os.path.exists("transactions.json"):
        with open("transactions.json", "r") as file:
            transactions_data = json.load(file)
        transactions_collection.delete_many({})
        transactions_collection.insert_many(transactions_data)
        print("Migrated transactions to MongoDB")

    # Migrate blockchain
    if os.path.exists("blockchain.json"):
        with open("blockchain.json", "r") as file:
            blockchain_data = json.load(file)
        blockchain_collection.delete_many({})
        blockchain_collection.insert_many(blockchain_data)
        print("Migrated blockchain to MongoDB")

    # Migrate offline transactions
    if os.path.exists("offline_transactions.txt"):
        offline_data = []
        with open("offline_transactions.txt", "r") as file:
            for line in file:
                if line.strip():
                    try:
                        from_wallet, to_wallet, amount = line.strip().split(",")
                        amount = int(amount)
                        offline_data.append({"from": from_wallet, "to": to_wallet, "amount": amount})
                    except (ValueError, IndexError):
                        continue
        offline_transactions_collection.delete_many({})
        offline_transactions_collection.insert_many(offline_data)
        print("Migrated offline transactions to MongoDB")

# Function to load users from MongoDB
def load_users():
    global users
    users = {}
    for user in users_collection.find():
        user_id = user["_id"]
        user.pop("_id")  # Remove MongoDB's _id field from the user dict
        users[user_id] = user
    for user_id, user in users.items():
        if "transaction_count" not in user:
            user["transaction_count"] = 0
            users_collection.update_one({"_id": user_id}, {"$set": {"transaction_count": 0}})
    train_face_recognizer()

# Function to save users to MongoDB
def save_users():
    for user_id, user in users.items():
        users_collection.update_one({"_id": user_id}, {"$set": user}, upsert=True)
    train_face_recognizer()

# Function to load transactions from MongoDB
def load_transactions():
    global transactions
    transactions = list(transactions_collection.find())
    for tx in transactions:
        tx.pop("_id", None)  # Remove MongoDB's _id field

# Function to save transactions to MongoDB
def save_transactions():
    transactions_collection.delete_many({})
    transactions_collection.insert_many(transactions)

# Function to load blockchain from MongoDB
def load_blockchain():
    global ledger
    ledger = list(blockchain_collection.find())
    for block in ledger:
        block.pop("_id", None)
    if not ledger:
        genesis_block = {
            "index": 0,
            "transaction": {"from": "system", "to": "system", "amount": 0, "online": True, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            "previous_hash": "0",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        genesis_block["hash"] = compute_hash(genesis_block)
        ledger = [genesis_block]
        save_blockchain()

# Function to save blockchain to MongoDB
def save_blockchain():
    blockchain_collection.delete_many({})
    blockchain_collection.insert_many(ledger)

# Function to load offline transactions from MongoDB
def load_offline_transactions():
    global offline_queue
    offline_queue = list(offline_transactions_collection.find())
    for tx in offline_queue:
        tx.pop("_id", None)

# Function to save offline transactions to MongoDB
def save_offline_transactions():
    offline_transactions_collection.delete_many({})
    offline_transactions_collection.insert_many(offline_queue)

# Function to add a transaction to the blockchain
def add_to_blockchain(transaction):
    previous_block = ledger[-1]
    new_block = {
        "index": len(ledger),
        "transaction": transaction,
        "previous_hash": previous_block["hash"],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    new_block["hash"] = compute_hash(new_block)
    ledger.append(new_block)
    save_blockchain()

# Simple PIN-based biometric authentication
def biometric_auth(user_id, entered_pin):
    user = users.get(user_id)
    if user and user["pin"] == entered_pin:
        return True
    output_messages.append(f"<span style='color: #ff6b6b'>✘ Biometric authentication failed for ID: {user_id}</span>")
    return False

# Register a new user
def register_user(initial_balance, pin, name, phone, email, face_images=None):
    new_id = random.randint(10000, 99999)
    while new_id in users:
        new_id = random.randint(10000, 99999)
    new_wallet = f"wallet_{new_id}"
    users[new_id] = {
        "wallet": new_wallet,
        "balance": initial_balance,
        "pin": pin,
        "name": name,
        "role": "user",
        "phone": phone,
        "email": email,
        "transaction_count": 0
    }
    save_users()

    if face_images:
        if len(face_images) != 3:
            return new_id
        valid_faces = save_face_images(new_id, face_images)
        if valid_faces < 3:
            print(f"Warning: Only {valid_faces} valid faces detected for user {name} (ID: {new_id}).")
        train_face_recognizer()

    return new_id

# Delete a user
def delete_user(user_id, pin):
    if biometric_auth(user_id, pin):
        if user_id in users:
            user_dir = os.path.join("faces", str(user_id))
            if os.path.exists(user_dir):
                import shutil
                shutil.rmtree(user_dir)
            del users[user_id]
            users_collection.delete_one({"_id": user_id})
            output_messages.append(f"<span style='color: #ff6b6b'>User ID {user_id} deleted successfully!</span>")
            train_face_recognizer()
        else:
            output_messages.append(f"<span style='color: #ff6b6b'>Error: User ID {user_id} not found!</span>")
    else:
        output_messages.append(f"<span style='color: #ff6b6b'>Error: Cannot delete user ID {user_id} - authentication failed!</span>")

# Reset transaction count for a user
def reset_transaction_count(user_id):
    if user_id in users:
        users[user_id]["transaction_count"] = 0
        save_users()
        output_messages.append(f"<span style='color: #06d6a0'>Transaction count reset for user ID {user_id} ({users[user_id]['name']})!</span>")
    else:
        output_messages.append(f"<span style='color: #ff6b6b'>Error: User ID {user_id} not found!</span>")

# Update user profile
def update_user_profile(user_id, current_pin, new_name, new_phone, new_email, new_pin):
    if not biometric_auth(user_id, current_pin):
        output_messages.append("<span style='color: #ff6b6b'>Error: Current PIN is incorrect!</span>")
        return False

    user = users.get(user_id)
    if not user:
        output_messages.append("<span style='color: #ff6b6b'>Error: User not found!</span>")
        return False

    user["name"] = new_name
    user["phone"] = new_phone
    user["email"] = new_email
    if new_pin:
        user["pin"] = new_pin
    save_users()
    output_messages.append(f"<span style='color: #06d6a0'>Profile updated successfully for {user['name']} (ID: {user_id})!</span>")
    return True

# Authenticate user
def authenticate_user(biometric_id):
    return users.get(biometric_id)

# Send payment with transaction limit
def send_payment(sender_id, receiver_id, sender_pin, receiver_pin, amount, is_online, is_admin=False):
    try:
        sender = authenticate_user(sender_id)
        receiver = authenticate_user(receiver_id)

        if sender is None or receiver is None:
            output_messages.append("<span style='color: #ff6b6b'>Error: One or both users not found!</span>")
            return

        if sender["wallet"] == receiver["wallet"]:
            output_messages.append("<span style='color: #ff6b6b'>Error: Cannot send payment to yourself!</span>")
            return

        if not biometric_auth(sender_id, sender_pin):
            return

        if is_admin and not biometric_auth(receiver_id, receiver_pin):
            return

        TRANSACTION_LIMIT = 3
        if sender["transaction_count"] >= TRANSACTION_LIMIT:
            sender["balance"] -= amount
            receiver["balance"] += amount
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            transaction = {
                "from": sender["wallet"],
                "to": receiver["wallet"],
                "amount": amount,
                "online": is_online,
                "time": timestamp,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "sender_balance": sender["balance"],
                "receiver_balance": receiver["balance"]
            }
            transactions.append(transaction)
            save_transactions()

            if is_online:
                add_to_blockchain(transaction)
                output_messages.append(f"<span style='color: #06d6a0'>💸 Sent {amount} Rs from {sender['name']} ({sender['wallet']}) to {receiver['name']} ({receiver['wallet']}) (Online) at {timestamp}</span>")
            else:
                offline_queue.append(transaction)
                output_messages.append(f"<span style='color: #06d6a0'>💸 Sent {amount} Rs from {sender['name']} ({sender['wallet']}) to {receiver['name']} ({receiver['wallet']}) (Offline - Queued) at {timestamp}</span>")

            time.sleep(1)
            sender["balance"] += amount
            receiver["balance"] -= amount
            save_users()
            output_messages.append(f"<span style='color: #ff6b6b'>⚠ Transaction limit reached for {sender['name']} (ID: {sender_id})! Transaction of {amount} Rs has been refunded.</span>")
            output_messages.append(f"<span style='color: #ade8f4'>New balance for {sender['name']} (ID: {sender_id}): {sender['balance']} Rs</span>")
            output_messages.append(f"<span style='color: #ade8f4'>New balance for {receiver['name']} (ID: {receiver_id}): {receiver['balance']} Rs</span>")

            refund_message_sender = f"Transaction Limit Reached: Your transaction of {amount} Rs to {receiver['name']} on {timestamp} was refunded. New balance: {sender['balance']} Rs."
            send_sms(sender["phone"], refund_message_sender)
            send_email(sender["email"], "Transaction Refunded", refund_message_sender)

            refund_message_receiver = f"Transaction Reversed: The transaction of {amount} Rs from {sender['name']} on {timestamp} was refunded to the sender. New balance: {receiver['balance']} Rs."
            send_sms(receiver["phone"], refund_message_receiver)
            send_email(receiver["email"], "Transaction Reversed", refund_message_receiver)
            return

        if sender["balance"] >= amount:
            sender["balance"] -= amount
            receiver["balance"] += amount
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            transaction = {
                "from": sender["wallet"],
                "to": receiver["wallet"],
                "amount": amount,
                "online": is_online,
                "time": timestamp,
                "sender_id": sender_id,
                "receiver_id": receiver_id,
                "sender_balance": sender["balance"],
                "receiver_balance": receiver["balance"]
            }
            transactions.append(transaction)
            save_transactions()
            
            if is_online:
                add_to_blockchain(transaction)
                output_messages.append(f"<span style='color: #06d6a0'>💸 Success! Sent {amount} Rs from {sender['name']} ({sender['wallet']}) to {receiver['name']} ({receiver['wallet']}) (Online) at {timestamp}</span>")
            else:
                offline_queue.append(transaction)
                output_messages.append(f"<span style='color: #06d6a0'>💸 Success! Sent {amount} Rs from {sender['name']} ({sender['wallet']}) to {receiver['name']} ({receiver['wallet']}) (Offline - Queued) at {timestamp}</span>")
            
            sender["transaction_count"] += 1
            output_messages.append(f"<span style='color: #ade8f4'>Transaction count for {sender['name']} (ID: {sender_id}): {sender['transaction_count']}/3</span>")
            output_messages.append(f"<span style='color: #ade8f4'>New balance for {sender['name']} (ID: {sender_id}): {sender['balance']} Rs</span>")
            output_messages.append(f"<span style='color: #ade8f4'>New balance for {receiver['name']} (ID: {receiver_id}): {receiver['balance']} Rs</span>")
            save_users()

            sender_message = f"Transaction Alert: You sent {amount} Rs to {receiver['name']} on {timestamp}. New balance: {sender['balance']} Rs. Transaction count: {sender['transaction_count']}/3."
            send_sms(sender["phone"], sender_message)
            send_email(sender["email"], "Transaction Alert", sender_message)

            receiver_message = f"Transaction Alert: You received {amount} Rs from {sender['name']} on {timestamp}. New balance: {receiver['balance']} Rs."
            send_sms(receiver["phone"], receiver_message)
            send_email(receiver["email"], "Transaction Alert", receiver_message)
        else:
            output_messages.append("<span style='color: #ff6b6b'>Error: Not enough funds!</span>")
    except Exception as e:
        print(f"Error in send_payment: {e}")
        raise

# Sync offline transactions
def sync_offline():
    global offline_queue
    if not offline_queue:
        output_messages.append("<span style='color: #90e0ef'>No offline transactions to sync!</span>")
        return

    for tx in offline_queue[:]:
        sender = next((u for u in users.values() if u["wallet"] == tx["from"]), None)
        receiver = next((u for u in users.values() if u["wallet"] == tx["to"]), None)
        if sender is None or receiver is None:
            output_messages.append(f"<span style='color: #ff6b6b'>Skipping transaction: Sender ({tx['from']}) or Receiver ({tx['to']}) not found!</span>")
            offline_queue.remove(tx)
            continue

        TRANSACTION_LIMIT = 3
        if sender["transaction_count"] >= TRANSACTION_LIMIT:
            output_messages.append(f"<span style='color: #ff6b6b'>⚠ Transaction limit reached for {sender['name']}! Offline transaction of {tx['amount']} Rs to {receiver['name']} skipped.</span>")
            offline_queue.remove(tx)
            continue

        sender_name = sender["name"]
        receiver_name = receiver["name"]
        tx["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tx["online"] = True
        tx["sender_id"] = next((uid for uid, u in users.items() if u["wallet"] == tx["from"]), None)
        tx["receiver_id"] = next((uid for uid, u in users.items() if u["wallet"] == tx["to"]), None)
        tx["sender_balance"] = sender["balance"]
        tx["receiver_balance"] = receiver["balance"]
        add_to_blockchain(tx)
        transactions.append(tx)
        offline_queue.remove(tx)
        sender["transaction_count"] += 1
        save_users()
        output_messages.append(f"<span style='color: #06d6a0'>Synced: {sender_name} ({tx['from']}) sent {tx['amount']} Rs to {receiver_name} ({tx['to']}) at {tx['time']}</span>")
        output_messages.append(f"<span style='color: #ade8f4'>Transaction count for {sender_name}: {sender['transaction_count']}/3</span>")

        sender_message = f"Transaction Synced: Your offline transaction of {tx['amount']} Rs to {receiver_name} was synced on {tx['time']}. Current balance: {sender['balance']} Rs."
        send_sms(sender["phone"], sender_message)
        send_email(sender["email"], "Transaction Synced", sender_message)

        receiver_message = f"Transaction Synced: You received {tx['amount']} Rs from {sender_name} on {tx['time']}. Current balance: {receiver['balance']} Rs."
        send_sms(receiver["phone"], receiver_message)
        send_email(receiver["email"], "Transaction Synced", receiver_message)

    save_offline_transactions()
    save_transactions()

# Show ledger (blockchain)
def show_ledger():
    output_messages.append("<span style='color: #90e0ef'>Transaction Ledger (Blockchain):</span>")
    if len(ledger) <= 1:
        output_messages.append("<span style='color: #ff6b6b'>No transactions yet!</span>")
    else:
        for block in ledger[1:]:
            tx = block["transaction"]
            sender = next((u for u in users.values() if u["wallet"] == tx["from"]), None)
            receiver = next((u for u in users.values() if u["wallet"] == tx["to"]), None)
            sender_name = sender["name"] if sender else "Unknown"
            receiver_name = receiver["name"] if receiver else "Unknown"
            output_messages.append(f"<span style='color: #ade8f4'>Block {block['index']}: {sender_name} ({tx['from']}) sent {tx['amount']} Rs to {receiver_name} ({tx['to']}) at {tx['time']}</span>")
            output_messages.append(f"<span style='color: #caf0f8'>  Hash: {block['hash']}</span>")
            output_messages.append(f"<span style='color: #caf0f8'>  Previous Hash: {block['previous_hash']}</span>")

# Check balance
def check_balance(biometric_id, pin):
    if biometric_auth(biometric_id, pin):
        user = authenticate_user(biometric_id)
        if user:
            output_messages.append(f"<span style='color: #ade8f4'>Balance for {user['name']} (ID: {biometric_id}, {user['wallet']}): {user['balance']} Rs</span>")
            output_messages.append(f"<span style='color: #ade8f4'>Transaction count for {user['name']}: {user['transaction_count']}/3</span>")
            balance_message = f"Balance Check: Your balance as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} is {user['balance']} Rs."
            send_sms(user["phone"], balance_message)
            send_email(user["email"], "Balance Check", balance_message)
        else:
            output_messages.append("<span style='color: #ff6b6b'>Error: User not found!</span>")

# Show transaction history
def show_transactions(user_id=None, search_query=None):
    if not transactions:
        return {"title": "Transaction History", "data": [{"message": "No transactions yet!"}]}

    filtered_transactions = transactions if user_id is None else [
        tx for tx in transactions
        if (tx.get("sender_id") == user_id or tx.get("receiver_id") == user_id)
    ]

    if search_query:
        search_query = search_query.lower()
        filtered_transactions = [
            tx for tx in filtered_transactions
            if (search_query in tx.get("from", "").lower() or
                search_query in tx.get("to", "").lower() or
                search_query in str(tx.get("amount", "")))
        ]

    if not filtered_transactions:
        return {"title": "Transaction History", "data": [{"message": "No transactions match your search!"}]}

    transaction_data = []
    for i, tx in enumerate(filtered_transactions):
        try:
            entry = {
                "id": i + 1,
                "from": tx.get("from", "Unknown"),
                "to": tx.get("to", "Unknown"),
                "amount": f"{tx.get('amount', 0)} Rs",
                "status": "Online" if tx.get("online", False) else "Offline",
                "time": tx.get("time", "Unknown"),
                "sender_balance": f"{tx.get('sender_balance', 0)} Rs",
                "receiver_balance": f"{tx.get('receiver_balance', 0)} Rs"
            }
            transaction_data.append(entry)
        except Exception as e:
            print(f"Error processing transaction {tx}: {e}")
            continue

    return {
        "title": "Transaction History",
        "data": transaction_data
    }

# Export transactions to CSV
def export_transactions():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "From", "To", "Amount", "Status", "Time", "Sender Balance", "Receiver Balance"])
    for i, tx in enumerate(transactions, 1):
        writer.writerow([
            i,
            tx.get("from", "Unknown"),
            tx.get("to", "Unknown"),
            f"{tx.get('amount', 0)} Rs",
            "Online" if tx.get("online", False) else "Offline",
            tx.get("time", "Unknown"),
            f"{tx.get('sender_balance', 0)} Rs",
            f"{tx.get('receiver_balance', 0)} Rs"
        ])
    output.seek(0)
    return output

# Routes
@app.route('/')
def index():
    global output_messages
    output_messages = []
    load_users()
    load_transactions()
    load_blockchain()
    load_offline_transactions()
    if "user_id" not in session:
        return render_template('login.html', users=users)
    user = users.get(session["user_id"])
    if not user:
        session.pop("user_id", None)
        return render_template('login.html', users=users)
    return render_template('index.html', messages=output_messages, users=users, current_user=user, current_user_id=session["user_id"])

@app.route('/login', methods=['GET', 'POST'])
def login():
    global output_messages
    output_messages = []
    load_users()
    if request.method == 'POST':
        auth_type = request.form.get('auth_type')
        user_id = request.form.get('user_id')
        pin = request.form.get('pin')

        if auth_type == 'pin':
            try:
                user_id = int(user_id)
                if biometric_auth(user_id, pin):
                    session["user_id"] = user_id
                    return render_template('index.html', messages=output_messages, users=users, current_user=users[user_id], current_user_id=user_id)
                else:
                    output_messages.append("<span style='color: #ff6b6b'>Login failed!</span>")
            except ValueError:
                output_messages.append("<span style='color: #ff6b6b'>Error: Invalid user ID!</span>")
    return render_template('login.html', messages=output_messages, users=users)

@app.route('/face_login', methods=['POST'])
def face_login():
    global output_messages
    output_messages = []
    if face_recognizer is None or face_cascade is None:
        return jsonify({"success": False, "message": "Face recognition is not available on this server."})

    if not is_face_recognizer_trained:
        return jsonify({"success": False, "message": "Face recognizer is not trained."})

    data = request.get_json()
    if not data or 'image' not in data or 'user_id' not in data:
        return jsonify({"success": False, "message": "Invalid request data!"})

    try:
        user_id = int(data.get('user_id'))
    except ValueError:
        return jsonify({"success": False, "message": "Invalid user ID!"})

    image_data = data.get('image')
    if not image_data or ',' not in image_data:
        return jsonify({"success": False, "message": "Invalid image data!"})

    try:
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            return jsonify({"success": False, "message": "Failed to decode image!"})

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=1, minSize=(15, 15))
        if len(faces) == 0:
            return jsonify({"success": False, "message": "No face detected!"})

        (x, y, w, h) = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (100, 100))

        label, confidence = face_recognizer.predict(face_roi)
        with open("label_map.json", "r") as f:
            label_map = json.load(f)

        if str(label) not in label_map:
            return jsonify({"success": False, "message": f"Predicted label {label} not found in label map!"})

        predicted_user_id = int(label_map.get(str(label)))
        if predicted_user_id == user_id and confidence < 150:
            session["user_id"] = user_id
            return jsonify({"success": True, "message": "Face recognition successful!"})
        else:
            return jsonify({"success": False, "message": f"Face recognition failed! Confidence: {confidence}"})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error processing image: {str(e)}"})

@app.route('/register', methods=['GET', 'POST'])
def register():
    global output_messages
    output_messages = []
    load_users()
    if request.method == 'POST':
        data = request.get_json()
        initial_balance = data.get('initial_balance')
        pin = data.get('pin')
        name = data.get('name')
        phone = data.get('phone')
        email = data.get('email')
        face_images = data.get('face_images')

        if not all([initial_balance, pin, name, phone, email]):
            return jsonify({"success": False, "message": "All fields are required!"})
        
        try:
            initial_balance = int(initial_balance)
            if initial_balance < 0:
                return jsonify({"success": False, "message": "Balance cannot be negative!"})
            if len(pin) < 4:
                return jsonify({"success": False, "message": "PIN must be at least 4 characters!"})
            if not phone.startswith('+') or len(phone) < 10:
                return jsonify({"success": False, "message": "Phone number must start with + and be at least 10 digits!"})
            if '@' not in email or '.' not in email:
                return jsonify({"success": False, "message": "Invalid email address!"})
            if face_images and len(face_images) != 3:
                return jsonify({"success": False, "message": "Exactly 3 face images are required!"})

            new_id = register_user(initial_balance, pin, name, phone, email, face_images)
            return jsonify({"success": True, "message": f"Registration successful! Your User ID is {new_id}."})
        except ValueError:
            return jsonify({"success": False, "message": "Invalid balance!"})
    return render_template('register.html', messages=output_messages, users=users)

@app.route('/logout')
def logout():
    session.pop("user_id", None)
    load_users()
    return render_template('login.html', users=users)

@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    global output_messages
    output_messages = []
    if "user_id" not in session:
        load_users()
        return render_template('login.html', messages=["<span style='color: #ff6b6b'>Please log in first!</span>"], users=users)

    user_id = session["user_id"]
    user = users.get(user_id)
    if not user:
        session.pop("user_id", None)
        return render_template('login.html', messages=["<span style='color: #ff6b6b'>Session expired, please log in again!</span>"], users=users)

    if request.method == 'POST':
        current_pin = request.form.get('current_pin')
        new_name = request.form.get('name')
        new_phone = request.form.get('phone')
        new_email = request.form.get('email')
        new_pin = request.form.get('new_pin')

        if not all([current_pin, new_name, new_phone, new_email]):
            output_messages.append("<span style='color: #ff6b6b'>Error: All fields except new PIN are required!</span>")
        elif not new_phone.startswith('+') or len(new_phone) < 10:
            output_messages.append("<span style='color: #ff6b6b'>Error: Phone number must start with + and be at least 10 digits!</span>")
        elif '@' not in new_email or '.' not in new_email:
            output_messages.append("<span style='color: #ff6b6b'>Error: Invalid email address!</span>")
        elif new_pin and len(new_pin) < 4:
            output_messages.append("<span style='color: #ff6b6b'>Error: New PIN must be at least 4 characters!</span>")
        else:
            update_user_profile(user_id, current_pin, new_name, new_phone, new_email, new_pin)
            return render_template('edit_profile.html', messages=output_messages, user=user)

    return render_template('edit_profile.html', messages=output_messages, user=user)

@app.route('/action', methods=['POST'])
def action():
    global output_messages
    output_messages = []
    if "user_id" not in session:
        load_users()
        return render_template('login.html', messages=["<span style='color: #ff6b6b'>Please log in first!</span>"], users=users)

    current_user = users.get(session["user_id"])
    if not current_user:
        session.pop("user_id", None)
        load_users()
        return render_template('login.html', messages=["<span style='color: #ff6b6b'>Session expired, please log in again!</span>"], users=users)
    
    action = request.form.get('action')
    if action == 'pay':
        if current_user["role"] in ["user", "admin"]:
            sender_id = request.form.get('sender_id')
            receiver_id = request.form.get('receiver_id')
            sender_pin = request.form.get('sender_pin')
            receiver_pin = request.form.get('receiver_pin')
            amount = request.form.get('amount')
            is_online = request.form.get('is_online') == 'yes'

            if not all([sender_id, receiver_id, sender_pin, amount]):
                output_messages.append("<span style='color: #ff6b6b'>Error: All fields are required!</span>")
            elif current_user["role"] == "admin" and not receiver_pin:
                output_messages.append("<span style='color: #ff6b6b'>Error: Receiver PIN is required for admin!</span>")
            else:
                try:
                    sender_id = int(sender_id)
                    receiver_id = int(receiver_id)
                    amount = int(amount)
                    if amount <= 0:
                        output_messages.append("<span style='color: #ff6b6b'>Error: Amount must be positive!</span>")
                    elif current_user["role"] == "user" and sender_id != session["user_id"]:
                        output_messages.append("<span style='color: #ff6b6b'>Error: Users can only send from their own account!</span>")
                    else:
                        send_payment(sender_id, receiver_id, sender_pin, receiver_pin, amount, is_online, is_admin=(current_user["role"] == "admin"))
                except ValueError:
                    output_messages.append("<span style='color: #ff6b6b'>Error: Invalid numeric input for IDs or amount!</span>")

    elif action == 'sync':
        sync_offline()

    elif action == 'ledger' and current_user["role"] == "admin":
        show_ledger()

    elif action == 'balance':
        if current_user["role"] in ["user", "admin"]:
            user_id = request.form.get('user_id')
            pin = request.form.get('pin')
            if not all([user_id, pin]):
                output_messages.append("<span style='color: #ff6b6b'>Error: User ID and PIN are required!</span>")
            else:
                try:
                    user_id = int(user_id)
                    if current_user["role"] == "user" and user_id != session["user_id"]:
                        output_messages.append("<span style='color: #ff6b6b'>Error: Users can only check their own balance!</span>")
                    else:
                        check_balance(user_id, pin)
                except ValueError:
                    output_messages.append("<span style='color: #ff6b6b'>Error: Invalid user ID!</span>")

    elif action == 'register' and current_user["role"] == "admin":
        initial_balance = request.form.get('initial_balance')
        pin = request.form.get('pin')
        name = request.form.get('name')
        phone = request.form.get('phone')
        email = request.form.get('email')
        if not all([initial_balance, pin, name, phone, email]):
            output_messages.append("<span style='color: #ff6b6b'>Error: Balance, PIN, Name, Phone, and Email are required!</span>")
        else:
            try:
                initial_balance = int(initial_balance)
                if initial_balance < 0:
                    output_messages.append("<span style='color: #ff6b6b'>Error: Balance cannot be negative!</span>")
                elif len(pin) < 4:
                    output_messages.append("<span style='color: #ff6b6b'>Error: PIN must be at least 4 characters!</span>")
                elif not phone.startswith('+') or len(phone) < 10:
                    output_messages.append("<span style='color: #ff6b6b'>Error: Phone number must start with + and be at least 10 digits!</span>")
                elif '@' not in email or '.' not in email:
                    output_messages.append("<span style='color: #ff6b6b'>Error: Invalid email address!</span>")
                else:
                    new_id = register_user(initial_balance, pin, name, phone, email)
                    output_messages.append(f"<span style='color: #06d6a0'>Registration successful! New User ID: {new_id}</span>")
            except ValueError:
                output_messages.append("<span style='color: #ff6b6b'>Error: Invalid balance!</span>")

    elif action == 'transactions':
        search_query = request.form.get('search_query', None)
        try:
            if current_user["role"] == "admin":
                transaction_data = show_transactions(search_query=search_query)
            else:
                transaction_data = show_transactions(session["user_id"], search_query)
            return render_template('result.html', messages=[], transaction_data=transaction_data, search_query=search_query)
        except Exception as e:
            output_messages.append(f"<span style='color: #ff6b6b'>Error loading transactions: {str(e)}</span>")
            return render_template('result.html', messages=output_messages)

    elif action == 'delete' and current_user["role"] == "admin":
        user_id = request.form.get('user_id')
        pin = request.form.get('pin')
        if not all([user_id, pin]):
            output_messages.append("<span style='color: #ff6b6b'>Error: User ID and PIN are required!</span>")
        else:
            try:
                user_id = int(user_id)
                delete_user(user_id, pin)
            except ValueError:
                output_messages.append("<span style='color: #ff6b6b'>Error: Invalid user ID!</span>")

    elif action == 'reset_transaction_count' and current_user["role"] == "admin":
        user_id = request.form.get('user_id')
        if not user_id:
            output_messages.append("<span style='color: #ff6b6b'>Error: User ID is required!</span>")
        else:
            try:
                user_id = int(user_id)
                reset_transaction_count(user_id)
            except ValueError:
                output_messages.append("<span style='color: #ff6b6b'>Error: Invalid user ID!</span>")

    elif action == 'export' and current_user["role"] == "admin":
        csv_data = export_transactions()
        return send_file(
            csv_data,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f"transactions_{int(datetime.now().timestamp())}.csv"
        )

    if action == 'balance':
        return render_template('result.html', messages=output_messages)
    return render_template('result.html', messages=output_messages)

if __name__ == '__main__':
    migrate_data_to_mongodb()  # Run migration once
    app.run(debug=True, host='127.0.0.1', port=5000)