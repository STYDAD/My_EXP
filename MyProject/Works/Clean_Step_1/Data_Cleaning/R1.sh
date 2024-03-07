#!/bin/bash
python3 WS14/w14.py &
echo "Data Cleaning start w14.py"

python3 WS15/w15.py &
echo "Data Cleaning start w15.py"

python3 WS16/w16.py &
echo "Data Cleaning start w16.py"

python3 WS17/w17.py &
echo "Data Cleaning start w17.py"

python3 WS18/w18.py &
echo "Data Cleaning start w18.py"

# python3 WS${19}/w${19}.py &
# echo "Data Cleaning start w${19}.py"

# python3 WS${20}/w${20}.py &
# echo "Data Cleaning start w${20}.py"

# python3 WS${21}/w${21}.py &
# echo "Data Cleaning start w${21}.py"

# python3 WS${22}/w${22}.py &
# echo "Data Cleaning start w${22}.py"

# python3 WS${23}/w${23}.py &
# echo "Data Cleaning start w${23}.py"



echo "All Workers End !!!"