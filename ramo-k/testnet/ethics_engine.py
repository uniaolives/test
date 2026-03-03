# ramo-k/testnet/ethics_engine.py
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/review', methods=['POST'])
def review():
    data = request.json

    # Article 3 check: Human veto override
    if data.get('human_veto'):
        return jsonify({'approved': False, 'reason': 'Art.3_human_veto'})

    # Article 6: Non-interference
    if data.get('affected_humans', 0) > 0 and not data.get('consent_verified'):
        return jsonify({'approved': False, 'reason': 'Art.6_consent_required'})

    # Default approval for mock
    return jsonify({
        'approved': True,
        'confidence': 0.95,
        'constitutional_basis': 'Art.6+EthicalConstraint'
    })

if __name__ == '__main__':
    print("Ramo K Ethics Engine starting on port 5000...")
    app.run(port=5000)
