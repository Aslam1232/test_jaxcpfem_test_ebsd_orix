cd ~/projets/jax-fem/applications/test_traction_simple/data/neper/traction_cube/

# Génère la tessellation pour 50 grains et un cube de 20 mm de côté
neper -T \
      -n 50 \
      -id 0 \
      -regularization 1 \
      -domain "cube(0.02,0.02,0.02)" \
      -format tess,ori,obj \
      -o domain50

# Maillage en éléments hexaédriques de premier ordre (HEX8)
neper -M domain50.tess \
      -elttype hex \
      -order 1 \
      -rcl 1 \
      -format msh \
      -o domain50

mv domain50.msh domain.msh
