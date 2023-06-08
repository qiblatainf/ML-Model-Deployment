from flask import Flask, jsonify, request
import sys

app = Flask(__name__)

from profanityfilter_detector import detect

@app.route('/textprofanity', methods = ['POST'])
async def textprofanity():
    if (request.method == 'POST'):
        text = request.get_json()['text']
        result = await detect(text)
    return jsonify({'Profanity Detected': result})
