#!/bin/bash
# deploy-maihh-connect.sh
echo "🦞 Deploying MaiHH Connect Agent Internet..."
docker-compose -f docker-compose.eternity-maihh.yml up -d
echo "✅ MaiHH Connect Deployed."
