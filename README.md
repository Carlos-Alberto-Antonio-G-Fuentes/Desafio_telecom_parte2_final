# Desafio_telecom_parte2_final
Challenge Data Science – Telecom X Parte2 y Final

1. Propósito del Análisis
El objetivo principal del análisis es predecir la cancelación de clientes (churn) en una empresa de telecomunicaciones. Para lograrlo, se aplicaron modelos de Machine Learning que permiten anticipar si un cliente va a cancelar el servicio en función de sus características y comportamiento.
Este tipo de análisis es fundamental para:
•	Detectar patrones asociados a la pérdida de clientes.
•	Identificar variables clave que influyen en la decisión de cancelar el servicio.
•	Optimizar estrategias de retención, enfocando recursos en los clientes con mayor riesgo de churn.
•	Evaluar y comparar el rendimiento de distintos modelos predictivos: Regresión Logística, Árbol de Decisión y Random Forest.

2. Proceso de Preparación de los Datos
a) Clasificación de las Variables
Durante la preparación de los datos, se realizó una separación clara entre:
•	Variables categóricas: aquellas que representan categorías o etiquetas. Ejemplos incluyen:
o	gender, Partner, Dependents, PhoneService, InternetService, Contract, PaymentMethod, entre otras.
•	Variables numéricas: aquellas que representan valores cuantitativos. Ejemplos incluyen:
o	tenure (tiempo de permanencia), MonthlyCharges, TotalCharges, Cuentas_Diarias.
Esta clasificación fue crucial para definir qué técnicas aplicar en la codificación y en los modelos.


b) Etapas de Normalización o Codificación
Dado que muchos modelos de ML no manejan directamente variables categóricas, se aplicaron transformaciones como:
•	Codificación Binaria para variables categóricas: Las variables categóricas fueron transformadas en variables numéricas (0 y Estas transformaciones aseguran que los modelos puedan interpretar correctamente la información.
c) Separación en Conjuntos de Entrenamiento y Prueba
Los datos fueron divididos en:
•	Conjunto de entrenamiento (X_train, y_train): usado para entrenar los modelos.
•	Conjunto de prueba (X_test, y_test): usado para evaluar su rendimiento en datos no vistos.
Esta división se realiza típicamente con train_test_split() de sklearn, asegurando que el modelo no memorice los datos y generalice bien.

d) Justificación de Decisiones en la Modelización
Se utilizaron tres modelos distintos para comparar su rendimiento:
1.	Regresión Logística:
o	Modelo lineal interpretable.
o	Útil para observar el peso de cada variable en la predicción de churn.
o	Adecuado como línea base.
2.	Árbol de Decisión (Decision Tree):
o	Captura relaciones no lineales.
o	Fácil de visualizar y entender.
o	Se puede regular con hiperparámetros como max_depth para evitar sobreajuste.
3.	Random Forest:
o	Ensamble de múltiples árboles.
o	Mejora la precisión y reduce el sobreajuste.
o	Recomendado para tareas donde se prioriza la exactitud y robustez del modelo.
Para evaluar estos modelos, se utilizó la matriz de confusión, lo que permite analizar visualmente:
•	Verdaderos positivos y negativos.
•	Falsos positivos y negativos.
Esto ayuda a entender no solo la precisión global, sino también qué tan bien el modelo identifica clientes con churn real.


3.	Ejemplos de gráficos e insights obtenidos durante el análisis exploratorio de datos (EDA).


	Variables Categóricas
a.- Perfil Demográfico

* SeniorCitizen: Los adultos mayores parecen tener una tasa de churn más alta que los no mayores. Esto podría reflejar una menor afinidad con ciertos servicios digitales o mayor sensibilidad al precio.

* Gender: No suele haber una gran diferencia entre hombres y mujeres en la tasa de abandono, lo que sugiere que el churn no está influido por el género.

* Partner: Clientes sin pareja o sin dependientes tienden a tener mayor churn, indicando que los usuarios con responsabilidades familiares están más vinculados a mantener sus servicios.

b.- Servicios contratados

* PhoneService y MultipleLines: Quienes no tienen servicio de teléfono o líneas múltiples muestran un churn superior. Tal vez se sientan menos comprometidos o utilizan alternativas externas.

* StreamingTV / StreamingMovies: Los usuarios sin servicios de streaming tienden a abandonar más, lo cual puede reflejar falta de uso o menor percepción de valor en el paquete contratado.

* OnlineSecurity / OnlineBackup / TechSupport / DeviceProtection: Los que no usan servicios adicionales de soporte o seguridad tienen tasas de churn significativamente mayores. Esto sugiere que los servicios complementarios ayudan a fidelizar.

3.- Uso de internet y contrato

* tiene_internet: Quienes no tienen internet casi no se quedan. Es el servicio central, y su ausencia es una señal clara de baja vinculación.

* contrato_1_ano_y_mas: Como suele suceder, los clientes con contratos largos tienen mucho menor churn. Contratos más extensos generan más compromiso.

* pago_automatico: Este método está asociado con menor churn. Posiblemente porque automatizar el pago ayuda a fidelizar al cliente.

4.- Facturación

* PaperlessBilling: Los clientes con facturación electrónica presentan ligeramente más churn. Esto podría estar correlacionado con perfiles más digitales, que también comparan y cambian más fácilmente de proveedor.

Conclusiones estratégicas
•	Invertir en servicios complementarios como respaldo, seguridad y soporte técnico ayuda a reducir la fuga de clientes.
•	Los contratos largos y el pago automático son poderosos mecanismos de retención.
•	Los clientes con menos vínculos personales o familiares (sin pareja ni dependientes) deberían ser segmentados para seguimiento personalizado.
•	Si se detectan perfiles sin uso de servicios clave como internet o streaming, pueden estar en riesgo de abandono.

 
	Variables Numéricas:

1.- tenure (antigüedad del cliente)

Los clientes con Churn=1, suelen irse temprano. Si alguien supera la barrera de los 30 meses, es mucho menos probable que se vaya. Esto puede marcar una zona crítica para retención.
Insight: Fidelizar desde el inicio puede reducir churn.
2.- MonthlyCharges (cargo mensual)
Interpretación: Los clientes que pagan más están más propensos a irse. Esto podría sugerir que hay sensibilidad al precio o que están pagando por servicios que no valoran completamente.
•	Si el total pagado es mucho menor en clientes que se van, hay un perfil de bajo valor que abandona más fácilmente.
•	Si las personas que abandonan pagan más en total, podría indicar sensibilidad al precio.
•	Si las personas que abandonan pagan más mensualmente, podría indicar sensibilidad al precio.
Insight:
a) MonthlyCharges funciona como un reflejo del ciclo de vida del cliente.
b) Cargos altos podrían generar insatisfacción si no hay percepción de valor.
c) Fomentar el uso de los servicios que paga el cliente, puede aumentar la retención.

Conclusión general:

-	La antigüedad del cliente (tenure) y su uso frecuente están fuertemente asociados con retención.
-	El precio puede ser un factor crítico para el churn.
-	Clientes nuevos con poca interacción y cargos altos están en zona de riesgo.
Se podría considerar planes escalonados o segmentación por valor.
Se debe detectar si el uso del servicio influye en el abandono. Por ejemplo, un cliente que paga mensualmente pero casi no accede a las plataforma podría mostrar baja interacción y estar en riesgo de abandono.
 
Se aplica VIF
El VIF (Variance Inflation Factor) nos dice cuánto se multiplica la varianza de una variable explicativa por la multicolinealidad. Idealmente, los valores deberían estar por debajo de 5, aunque algunos toleran hasta 10. Multicolinealidad extrema (inf): probablemente es combinación lineal exacta con otra variable, por lo cual solo dejaremos las variables MonthlyCharges y tenure y eliminaremos las variables Cuentas_Diarias y TotalCharges por redundancia en nuestro modelo.
Const es el valor, de la constante del modelo.

Conclusión de Modelos: 

El modelo de Random Forest es el mejor de los tres porque:

Tiene mayor precisión para predecir churn (0.69).

Tiene el mayor accuracy general (0.81).

Tiene un recall similar a Regresión Logística, pero con menos falsos positivos.

Random Forest ofrece el mejor balance entre identificar correctamente a los clientes que harán churn sin comprometer demasiado los falsos positivos. Por tanto, es el más recomendable.

Bibioteca utilizada:

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches 
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score,recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

Otras instrucciones y especificaciones se encuentran el en cuaderno de google colab
