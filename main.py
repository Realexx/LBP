import cv2
import numpy as np
import matplotlib.pyplot as plt


##### LBP #####
# Fonction pour calculer le motif binaire local (LBP) d'un seul pixel de l'image
def calculer_lbp_pixel(image, x, y):
    centre = image[x, y]  # Valeur du pixel central
    code = 0  # Initialisation du motif binaire local
    # Pour les 8 pixels autour du pixel central
    for i, j in [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]:
        voisin = image[x + i, y + j]  # Valeur du pixel voisin
        code <<= 1  # Décalage de bit vers la gauche pour préparer la prochaine comparaison
        if voisin <= centre:  # Si la valeur du voisin est inférieur ou égale au centre, mettre le bit à '1'
            code |= 1
    return code  # Retourne le motif binaire local calculé


# Fonction pour calculer l'image LBP complète
def calculer_lbp_image(image):
    hauteur, largeur = image.shape  # Récupération des dimensions de l'image
    image_lbp = np.zeros((hauteur, largeur), dtype=np.uint8)  # Initialisation de l'image LBP
    # Parcourir l'image sauf la bordure
    for i in range(1, hauteur - 1):
        for j in range(1, largeur - 1):
            image_lbp[i, j] = calculer_lbp_pixel(image, i, j)  # Calcul du LBP pour chaque pixel de l'image
    return image_lbp  # Retourne l'image LBP calculée


# Lecture de l'image
image = cv2.imread("1.4.10.tiff", cv2.IMREAD_GRAYSCALE)

# Conversion en niveaux de gris si l'image n'est pas déjà en niveaux de gris
if len(image.shape) > 2:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calcul du LBP pour l'image donnée
image_lbp = calculer_lbp_image(image)

# Affichage de l'image originale
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Image originale")
plt.axis('off')

# Affichage de l'histogramme de l'image originale
plt.subplot(2, 2, 2)
plt.hist(image.ravel(), 256, [0, 256], color='black')
plt.title("Histogramme de l'image originale")
plt.xlabel("Niveau de gris")
plt.ylabel("Nombre de pixels")

# Affichage de l'image LBP
plt.subplot(2, 2, 3)
plt.imshow(image_lbp, cmap='gray')
plt.title("Image LBP")
plt.axis('off')

# Affichage de l'histogramme de l'image LBP
plt.subplot(2, 2, 4)
plt.hist(image_lbp.ravel(), 256, [0, 256], color='black')
plt.title("Histogramme de l'image LBP")
plt.xlabel("Valeur des pixels")
plt.ylabel("Nombre d'occurrences")

plt.tight_layout()
plt.show()


##### MEAN-LBP #####
# Fonction pour calculer le Mean LBP d'un pixel
def calculer_mean_lbp_pixel(image, x, y):
    centre = image[x, y]
    somme_voisins = 0
    for i, j in [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]:
        somme_voisins += image[x + i, y + j]  # Somme des pixels voisins
    moyenne_voisins = somme_voisins // 8
    if centre >= moyenne_voisins:
        return 1
    else:
        return 0


# Fonction pour calculer l'image Mean LBP complète
def calculer_mean_lbp_image(image):
    hauteur, largeur = image.shape  # Récupération des dimensions de l'image
    image_mean_lbp = np.zeros((hauteur, largeur), dtype=np.uint8)  # Initialisation de l'image Mean LBP
    # Parcourir l'image
    for i in range(1, hauteur - 1):
        for j in range(1, largeur - 1):
            # Calcul du Mean LBP pour chaque pixel de l'image
            image_mean_lbp[i, j] = calculer_mean_lbp_pixel(image, i, j)
    return image_mean_lbp  # Retourne l'image Mean LBP calculée


# Lecture de l'image
image = cv2.imread("1.4.10.tiff", cv2.IMREAD_GRAYSCALE)

if len(image.shape) > 2:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calcul du Mean LBP pour l'image donnée
image_mean_lbp = calculer_mean_lbp_image(image)

# Affichage de l'image Mean LBP
plt.subplot(1, 2, 1)
plt.imshow(image_mean_lbp, cmap='gray')
plt.title("Image Mean LBP")
plt.axis('off')

# Affichage de l'histogramme de l'image Mean LBP
plt.subplot(1, 2, 2)
plt.hist(image_mean_lbp.ravel(), 256, [0, 256], color='black')
plt.title("Histogramme de l'image Mean LBP")
plt.xlabel("Valeur des pixels")
plt.ylabel("Nombre d'occurrences")

plt.tight_layout()
plt.show()
