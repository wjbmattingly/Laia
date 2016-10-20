#!/bin/bash
set +e;  # We want to run all tests, even if some fail.

# This script should run from the main Laia directory, cd to there.
EXPECTED_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)";
cd "$EXPECTED_DIR";

# th is required, exit if not present.
which th || exit 1;

# ADD YOUR TESTS HERE
th test/nn/ImageColumnSequence.lua;
th test/util/math.lua;
th test/util/table.lua;
