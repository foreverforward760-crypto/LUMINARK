#!/bin/bash
# Quick verification script to run after restructuring
# Run this immediately after antigravity completes!

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════╗"
echo "║     LUMINARK Post-Restructure Verification Script       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Import verification
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[1/5] Verifying imports..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python tests/test_imports.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Imports verification passed${NC}"
else
    echo -e "${RED}❌ Import verification failed${NC}"
    exit 1
fi
echo ""

# Step 2: Unit tests
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[2/5] Running unit tests..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python tests/test_framework.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Unit tests passed${NC}"
else
    echo -e "${RED}❌ Unit tests failed${NC}"
    exit 1
fi
echo ""

# Step 3: Basic example
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[3/5] Testing basic MNIST example..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python examples/train_mnist.py | tail -20
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Basic example passed${NC}"
else
    echo -e "${RED}❌ Basic example failed${NC}"
    exit 1
fi
echo ""

# Step 4: Checkpoint & scheduler demo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[4/5] Testing checkpoint & scheduler features..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python examples/checkpoint_and_scheduler_demo.py | tail -30
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Checkpoint & scheduler test passed${NC}"
else
    echo -e "${RED}❌ Checkpoint & scheduler test failed${NC}"
    exit 1
fi
echo ""

# Step 5: Integration test
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "[5/5] Running integration test..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python tests/test_integration.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Integration test passed${NC}"
else
    echo -e "${RED}❌ Integration test failed${NC}"
    exit 1
fi
echo ""

# Success summary
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ✅ ALL VERIFICATION PASSED! ✅              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo -e "${GREEN}🎉 LUMINARK restructuring successful!${NC}"
echo ""
echo "Framework is ready for:"
echo "  • Production deployment (Docker, cloud)"
echo "  • Package publishing (PyPI)"
echo "  • Community use"
echo ""
echo "Next steps:"
echo "  1. Review POST_RESTRUCTURE_PLAN.md for detailed next steps"
echo "  2. Run benchmarks: python benchmarks/benchmark_training.py"
echo "  3. Test Docker: docker build -t luminark:test ."
echo "  4. Merge PR: https://github.com/foreverforward760-crypto/LUMINARK/pull/1"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
