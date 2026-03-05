# scripts/materialize.sh
echo "🜁 Starting Arkhe(n) Materialization..."
docker-compose -f docker-compose.synthesis.yml build
docker-compose -f docker-compose.synthesis.yml up -d
echo "✅ Organism Operational."
