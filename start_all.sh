#!/bin/bash

echo "🚀 Starting Fitness AI Trainer (Next.js + FastAPI)"
echo "=================================================="

# Check if backend directory exists
if [ ! -d "backend" ]; then
    echo "❌ Error: backend directory not found"
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo "❌ Error: frontend directory not found"
    exit 1
fi

# Start backend in background
echo ""
echo "1️⃣ Starting Python FastAPI Backend (Port 8000)..."
cd backend

# Activate virtual environment if exists
if [ -d "../venv_mediapipe" ]; then
    source ../venv_mediapipe/bin/activate
fi

# Install backend dependencies
pip install -q -r requirements.txt

# Start backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "✅ Backend started (PID: $BACKEND_PID)"

cd ..

# Wait for backend to be ready
echo ""
echo "⏳ Waiting for backend to be ready..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/ > /dev/null; then
    echo "✅ Backend is ready"
else
    echo "⚠️  Backend may not be ready yet, but continuing..."
fi

# Start frontend
echo ""
echo "2️⃣ Starting Next.js Frontend (Port 3000)..."
cd frontend

# Install frontend dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start frontend
npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend started (PID: $FRONTEND_PID)"

cd ..

# Display info
echo ""
echo "=================================================="
echo "✅ Both servers are running!"
echo ""
echo "📡 Backend API:  http://localhost:8000"
echo "🌐 Frontend App: http://localhost:3000"
echo ""
echo "📖 API Docs:     http://localhost:8000/docs"
echo ""
echo "=================================================="
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Save PIDs for cleanup
echo $BACKEND_PID > .backend.pid
echo $FRONTEND_PID > .frontend.pid

# Wait for user to stop
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; rm -f .backend.pid .frontend.pid; echo '\n\n🛑 Servers stopped'; exit" INT TERM

# Keep script running
wait
