
OUTPUT_DIR="./notebook2py"
NOTEBOOK_DIR="../notebooks"
#NOTEBOOK_FILES=`ls $NOTEBOOK_DIR/*.ipynb`
NOTEBOOK_FILES="$NOTEBOOK_DIR/visualizando.dados.ipynb"

echo "[ FILES ] "
echo $NOTEBOOK_FILES
echo "----------------------"

for f in $NOTEBOOK_FILES; do
    echo "[ `basename $f` ]"
    jupyter.tgdesenv nbconvert --to script --output-dir=$OUTPUT_DIR $f
    echo "[ FIM ]"
done


doxygen 

