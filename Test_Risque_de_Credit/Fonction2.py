#import chardet
import pandas as pd
from pandas import DataFrame

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import seaborn as sns
import math
import shap
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, f1_score, recall_score, precision_score, mean_absolute_error, log_loss
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from skopt import BayesSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from scipy.stats import pearsonr, spearmanr, kruskal, mannwhitneyu, f_oneway, ttest_ind, chi2_contingency
import statsmodels.api as sm
import joblib


###########################################################################
#                       CHARGEMENT ET PRÉPARATION DES DONNÉES
#########################################################################


# ChargementDonnees
class ChargementDonnees:
    """
    Classe pour le chargement et le prétraitement des données d'un fichier CSV.
    
    Attributes:
        url (str): Lien vers le fichier CSV contenant les données.
        sep (str): Séparateur utilisé dans le fichier CSV, par défaut un espace.
        encoding (str): Encodage du fichier CSV, par défaut "utf-8".
        df (DataFrame): DataFrame contenant les données importées.
    """

    def __init__(self, url, sep=" ", encoding="utf-8"):
        """
        Initialise une instance de la classe ChargementDonnees avec le lien du fichier, 
        le séparateur, et l'encodage.

        Args:
            url (str): Lien vers le fichier CSV.
            sep (str): Séparateur dans le fichier CSV, par défaut un espace.
            encoding (str): Encodage du fichier CSV, par défaut "utf-8".
        """
        self.url = url
        self.sep = sep
        self.encoding = encoding
        self.df = None

    def importation_donnees(self):
        """
        Importe les données à partir du fichier CSV et les stocke dans un DataFrame.
        
        Définie les noms de colonnes pour le dataset German Credit. En cas de succès, 
        le DataFrame est affiché avec le nombre de lignes et de colonnes.

        Returns:
            DataFrame: DataFrame contenant les données importées.
        """
        # Colonnes pour le dataset German Credit
        columns = [
            'statut_du_compte_chèque_existant',
            'durée_en_mois',
            'historique_de_crédit',
            'objectif',
            'montant_du_crédit',
            'compte_épargne_obligations',
            'ancienneté_dans_emploi_actuel',
            'taux_déchéance_en_pourcentage_du_revenu_disponible',
            'statut_personnel_et_sexe',
            'autres_débiteurs_garanties',
            'ancienneté_de_la_résidence_actuelle',
            'propriété',
            'age',
            'autres_plans_de_remboursement',
            'logement',
            'nombre_de_crédits_existants_dans_cette_banque',
            'emploi',
            'nombre_de_personnes_à_charge',
            'téléphone',
            'travailleur_étranger',
            'target'
        ]

        try:
            # Charger les données dans un DataFrame
            self.df = pd.read_csv(self.url, sep=self.sep, encoding=self.encoding, names=columns)
            print(f"Le DataFrame contient {self.df.shape[0]} lignes et {self.df.shape[1]} colonnes.")
        except FileNotFoundError:
            print("Erreur : Le fichier spécifié est introuvable.")
        return self.df

    def traitement_initiale(self):
        """
        Applique le traitement initial aux données en utilisant des dictionnaires de mappage.
        
        Les colonnes qualitatives sont mappées en valeurs plus compréhensibles et la 
        variable 'target' est convertie en binaire pour la classification.
        
        Returns:
            DataFrame: DataFrame avec les valeurs mappées.
        """
        # Dictionnaires de mappage pour les colonnes qualitatives
        mapping_dict = {
            "statut_du_compte_chèque_existant": {
                "A11": "<0DM", "A12": "0<=x<200DM", "A13": ">=200DM/salaires_assignés_pendant_au_moins_1_an", "A14": "pas_de_compte_chèque"
            },
            "historique_de_crédit": {
                "A30": "aucun_crédit_pris/tous_les_crédits_remboursés_correctement", "A31": "tous_les_crédits_à_cette_banque_remboursés_correctement",
                "A32": "crédits_existants_remboursés_correctement_jusquà_présent", "A33": "retard_dans_le_remboursement_par_le_passé",
                "A34": "compte_critique/autres_crédits_existants_(pas_à_cette_banque)"
            },
            "objectif": {
                "A40": "voiture(neuve)", "A41": "voiture(doccasion)", "A42": "meubles/équipements", "A43": "radio/télévision",
                "A44": "appareils_ménagers", "A45": "réparations", "A46": "éducation", "A47": "(vacances-n_existepas?)",
                "A48": "reconversion", "A49": "affaires", "A410": "autres"
            },
            "compte_épargne_obligations": {
                "A61": "<100DM", "A62": "100<=x<500DM", "A63": "500<=x<1000DM", "A64": ">=1000DM", "A65": "inconnu/pas_de_compte_épargne"
            },
            "ancienneté_dans_emploi_actuel": {
                "A71": "sans emploi", "A72": "<1_an", "A73": "1<=x<4_ans", "A74": "4<=x<7_ans", "A75": ">=7_ans"
            },
            "statut_personnel_et_sexe": {
                "A91": "homme:divorcé/séparé", "A92": "femme:divorcée/séparée/mariée", "A93": "homme:célibataire",
                "A94": "homme:marié/veuf", "A95": "femme:célibataire"
            },
            "autres_débiteurs_garanties": {
                "A101": "aucun", "A102": "co-emprunteur", "A103": "garant"
            },
            "propriété": {
                "A121": "immobilier", "A122": "contrat_dépargne/logement_ou_assurance_vie",
                "A123": "voiture_ou_autre/non_inclus_dans_lattribut_6", "A124": "inconnu/pas_de_propriété"
            },
            "autres_plans_de_remboursement": {
                "A141": "banque", "A142": "magasins", "A143": "aucun"
            },
            "logement": {
                "A151": "location", "A152": "propriétaire", "A153": "gratuit"
            },
            "emploi": {
                "A171": "sans_emploi/non_qualifié_non_résident", "A172": "non_qualifié_résident", "A173": "employé_qualifié/fonctionnaire", "A174": "cadre/indépendant/employé_hautement_qualifié/officier"
            },
            "téléphone": {
                "A191": "aucun", "A192": "oui_enregistré_au_nom_du_client"
            },
            "travailleur_étranger": {
                "A201": "oui", "A202": "non"
            },
            "target": {1: 0, 2: 1}
        }

        # Applique le mappage sur chaque colonne qualitative
        for column, mapping in mapping_dict.items():
            if column in self.df.columns:
                self.df[column] = self.df[column].map(mapping)
        print("Le traitement initial a été appliqué.")
        return self.df



###########################################################################
#                       Analyse exploratoire
###########################################################################


class EDA:
    def __init__(self, df=None):
        """
        Initialise la classe EDA avec un DataFrame optionnel.
        """
        self.df = df
    
    def description(self, df=None, colonnes_specifiques=[
        'durée_en_mois', 
        'montant_du_crédit', 
        'taux_déchéance_en_pourcentage_du_revenu_disponible', 
        'age', 
        'nombre_de_crédits_existants_dans_cette_banque', 
        'nombre_de_personnes_à_charge', 
        'ancienneté_de_la_résidence_actuelle'
    ]):
        """
        Affiche une description complète du DataFrame, y compris :
        - Nombre de lignes et colonnes
        - Types de données pour chaque variable
        - Statistiques descriptives pour les colonnes spécifiées
        """
        # Utiliser le DataFrame passé en paramètre ou celui de l'instance
        df = df if df is not None else self.df
        
        if df is not None:
            # Nombre de lignes et de colonnes
            row, col = df.shape
            print(f"Le DataFrame contient {col} variables et {row} observations.\n")
            
            # Types de données pour chaque variable
            print("Types de données des variables :\n")
            print(df.dtypes)
            
            # Statistiques descriptives pour les colonnes spécifiées
            print("\nStatistiques descriptives pour les colonnes spécifiées :\n")
            colonnes_a_decrire = [col for col in colonnes_specifiques if col in df.columns]
            if colonnes_a_decrire:
                stats_specifiques = df[colonnes_a_decrire].describe().T
                print(stats_specifiques)
            else:
                print("Aucune des colonnes spécifiées n'existe dans le DataFrame.")
        else:
            print("Aucun DataFrame n'a été fourni.")
            
    def identifier_doublons_constantes(self, df=None):
        """
        Identifie les doublons dans le DataFrame et les colonnes avec des valeurs constantes.
        """
        df = df if df is not None else self.df

        if df is not None:
            valeurs_constantes = [var for var in df.columns if df[var].nunique() <= 1]
            print(f"Colonnes avec des valeurs constantes: {valeurs_constantes}")

            nb_doublons = df.duplicated().sum()
            print(f"Nombre de doublons dans le DataFrame: {nb_doublons}")
            return valeurs_constantes, nb_doublons
        else:
            print("Aucune donnée n'a été importée.")
            return None, None

    def afficher_valeurs_manquantes(self, df=None):
        """
        Identifie et affiche le nombre total de lignes avec des valeurs manquantes dans le DataFrame
        et affiche les informations sur les valeurs manquantes sous forme de tableau.
        """
        df = df if df is not None else self.df

        if df is not None:
            total_lignes_manquantes = df[df.isnull().any(axis=1)].shape[0]
            print(f"Nombre total de lignes avec des valeurs manquantes dans le DataFrame: {total_lignes_manquantes}")
            resultats = pd.DataFrame(df.dtypes, columns=['Type']).reset_index().rename(columns={'index': 'Variable'})
            resultats['Nombre des valeurs manquantes'] = df.isnull().sum().values
            resultats['Taux de valeurs manquantes (%)'] = (df.isnull().sum().values / len(df)) * 100
            resultats.sort_values(by='Nombre des valeurs manquantes', ascending=False, inplace=True)
            return resultats
        else:
            print("Aucune donnée n'a été importée.")
            return None

    def afficher_lignes_manquantes_target(self, df=None, colonne_cible='target', valeur=1):
        """
        Sélectionne et affiche uniquement le nombre total de lignes qui ont des valeurs manquantes et où 
        la colonne cible est égale à la valeur spécifiée.
        Arguments:
        - colonne_cible: nom de la colonne sur laquelle appliquer le filtre (par défaut 'embauche')
        - valeur: valeur spécifique à filtrer (par défaut 1)
        """
        df = df if df is not None else self.df

        if df is not None:
            total_lignes_manquantes_cible = df[df.isnull().any(axis=1) & (df[colonne_cible] == valeur)].shape[0]
            print(f"Nombre total de lignes avec des valeurs manquantes où '{colonne_cible}' = {valeur}: {total_lignes_manquantes_cible}")
            return total_lignes_manquantes_cible
        else:
            print("Aucune donnée n'a été importée.")
            return None


###########################################################################
#                       Data Visualisation 
###########################################################################

# Classe EDA_Visualizer
class EDA_Visualizer:
    def __init__(self, df_list, palette=None):
        """
        Initialise la classe avec une liste de DataFrames et une palette de couleurs personnalisée.
        """
        if not isinstance(df_list, list):
            df_list = [df_list]
        self.df_list = df_list
        self.palette = palette if palette else [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2',
            '#7f7f7f', '#bcbd22', '#17becf', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5', '#aec7e8'
        ]

    def generate_color_palette(self, df, col):
        """
        Génère une palette de couleurs fixe pour chaque catégorie dans une colonne.
        """
        unique_values = df[col].dropna().unique()
        return {value: self.palette[i % len(self.palette)] for i, value in enumerate(unique_values)}

    def general_plot_pie_charts(self, compare_column=None, categorical_columns=None, label_df1='DF1', label_df2='DF2', show_percentage=True):
        """
        Génère des diagrammes circulaires pour les colonnes catégorielles spécifiées avec des options de comparaison.
        """
        if categorical_columns is None:
            categorical_columns = self.df_list[0].select_dtypes(include=['object']).columns.tolist()
        
        df1 = self.df_list[0]
        df2 = self.df_list[1] if len(self.df_list) > 1 else None

        def autopct_format(values):
            def _autopct(pct):
                total = sum(values)
                val = int(round(pct * total / 100.0))
                return f"{pct:.1f}%" if show_percentage else f"{val}"
            return _autopct

        if compare_column:
            unique_categories = df1[compare_column].unique()
            n_categories = len(unique_categories)
            n_cols = n_categories
            n_rows = len(categorical_columns)
            figsize = (n_cols * 5, n_rows * 5)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

            for i, col in enumerate(categorical_columns):
                color_mapping = self.generate_color_palette(df1, col)
                for j, category in enumerate(unique_categories):
                    data = df1[df1[compare_column] == category][col].value_counts()
                    colors = [color_mapping[value] for value in data.index]
                    ax = axes[i, j]
                    ax.pie(data, labels=data.index, colors=colors, autopct=autopct_format(data), startangle=90)
                    ax.axis('equal')
                    ax.set_title(f'{category} - {col}', fontsize=8)
                    
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.6, hspace=1)
            plt.show()
        
        else:
            n_dfs = 2 if df2 is not None else 1
            n_cols = 2
            n_rows = math.ceil(len(categorical_columns) * n_dfs / n_cols)
            figsize = (22, n_rows * 5)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
            axes = axes.flatten()

            for i, col in enumerate(categorical_columns):
                color_mapping = self.generate_color_palette(df1, col)
                data_df1 = df1[col].value_counts()
                colors_df1 = [color_mapping[value] for value in data_df1.index]
                axes[i * n_dfs].pie(data_df1, labels=data_df1.index, colors=colors_df1, autopct=autopct_format(data_df1), startangle=90)
                axes[i * n_dfs].axis('equal')
                axes[i * n_dfs].set_title(f'{label_df1} - {col}', fontsize=16)
                
                if df2 is not None:
                    data_df2 = df2[col].value_counts()
                    colors_df2 = [color_mapping[value] for value in data_df2.index]
                    axes[i * n_dfs + 1].pie(data_df2, labels=data_df2.index, colors=colors_df2, autopct=autopct_format(data_df2), startangle=90)
                    axes[i * n_dfs + 1].axis('equal')
                    axes[i * n_dfs + 1].set_title(f'{label_df2} - {col}', fontsize=16)

            for ax in axes[len(categorical_columns) * n_dfs:]:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

    def compare_distributions(self, quanti_cols, compare_column=None, palette=('blue', 'red'), figsize=(12, 8)):
        """
        Compare les distributions de variables quantitatives entre groupes ou DataFrames.
        
        Arguments :
        - quanti_cols : Liste des colonnes quantitatives à comparer.
        - compare_column : Colonne pour diviser les données en groupes (par ex. 'target').
        - palette : Couleurs pour distinguer les groupes (par défaut 'blue' pour le groupe 1 et 'red' pour le groupe 2).
        - figsize : Taille de la figure générée.
        """
        n_dfs = len(self.df_list)
        n_cols = 2
        n_rows = (len(quanti_cols) + 1) // 2 * n_dfs
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for idx, df in enumerate(self.df_list):
            for i, col in enumerate(quanti_cols):
                plot_idx = idx * len(quanti_cols) + i

                if compare_column and compare_column in df.columns:
                    unique_values = df[compare_column].dropna().unique()
                    for j, value in enumerate(unique_values[:2]):
                        sns.histplot(
                            df[df[compare_column] == value][col],
                            kde=True,
                            color=palette[j % len(palette)],
                            ax=axes[plot_idx],
                            label=f"{compare_column} = {value}"
                        )
                    axes[plot_idx].legend()
                    axes[plot_idx].set_title(f'Distribution de {col} par {compare_column} - Dataset {idx + 1}')
                else:
                    sns.histplot(df[col], kde=True, color=palette[0], ax=axes[plot_idx])
                    axes[plot_idx].set_title(f'Distribution de {col} - Dataset {idx + 1}')

        for ax in axes[len(quanti_cols) * n_dfs:]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    def plot_box_by_category(self, df_index, x_column, y_column, hue_column=None, title='', xlabel='', ylabel='', source='', rotation=45, figsize=(20, 10)):
        """
        Trace un boxplot basé sur des colonnes spécifiques du DataFrame sélectionné.

        Arguments :
        - df_index : L'indice du DataFrame dans la liste self.df_list.
        - x_column : Nom de la colonne pour l'axe des X (catégorielle)
        - y_column : Nom de la colonne pour l'axe des Y (quantitative)
        - hue_column : (Optionnel) Nom de la colonne pour ajouter une distinction par couleur (catégorielle)
        - title : Titre du graphique
        - xlabel : Nom à afficher sur l'axe X
        - ylabel : Nom à afficher sur l'axe Y
        - source : Texte source à afficher au-dessus du graphique
        - rotation : Rotation des ticks sur l'axe X (par défaut à 45°)
        - figsize : Taille de la figure (par défaut à (20, 10))

        Retour :
        - Affiche un graphique en boîte basé sur les colonnes spécifiées.
        """
        df = self.df_list[df_index]

        sns.catplot(x=x_column, y=y_column, hue=hue_column, kind="box", data=df, height=figsize[1], aspect=figsize[0] / figsize[1])
        plt.xlabel(xlabel if xlabel else x_column, fontsize=15)
        plt.ylabel(ylabel if ylabel else y_column, fontsize=15)
        plt.title(title, fontsize=20)
        if source:
            plt.suptitle(f'Source : {source}', fontsize=15, x=0.5, y=0.95)
        plt.xticks(rotation=rotation)
        plt.show()

    def create_boxplots_for_categories(self, df_index, categorical_vars, numerical_vars):
        """
        Crée des boxplots pour les variables catégorielles et numériques spécifiées dans un DataFrame sélectionné.

        Arguments :
        - df_index : L'indice du DataFrame dans la liste self.df_list.
        - categorical_vars : Liste des variables catégorielles.
        - numerical_vars : Liste des variables numériques.
        
        Retour :
        - Affiche des boxplots pour chaque combinaison de variable catégorielle et numérique spécifiée.
        """
        df = self.df_list[df_index]
        
        for categorical_var in categorical_vars:
            plt.figure(figsize=(15, 12))
            
            for i, numerical_var in enumerate(numerical_vars, 1):
                if categorical_var in df.columns and numerical_var in df.columns:
                    plt.subplot(2, 2, i)
                    sns.boxplot(x=categorical_var, y=numerical_var, data=df)
                    plt.title(f'Boxplot de {numerical_var} par {categorical_var}')
                    plt.xlabel(categorical_var)
                    plt.ylabel(numerical_var)
                    plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()

    def plot_correlation_heatmaps(self, quanti_cols, compare_column=None, method='spearman', figsize=(12, 10)):
        """
        Génère des heatmaps agrandis pour les matrices de corrélation.
        """
        if quanti_cols is None:
            raise ValueError("Veuillez spécifier des colonnes quantitatives pour la corrélation.")

        if compare_column:
            group1 = self.df_list[0][self.df_list[0][compare_column] == self.df_list[0][compare_column].unique()[0]]
            group2 = self.df_list[0][self.df_list[0][compare_column] == self.df_list[0][compare_column].unique()[1]]
            
            corr1 = group1[quanti_cols].corr(method=method)
            corr2 = group2[quanti_cols].corr(method=method)
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            sns.heatmap(corr1, vmin=-1, vmax=1, annot=True, cmap='coolwarm', ax=axes[0])
            axes[0].set_title(f'{compare_column} = {group1[compare_column].unique()[0]} ({method.capitalize()})')
            sns.heatmap(corr2, vmin=-1, vmax=1, annot=True, cmap='coolwarm', ax=axes[1])
            axes[1].set_title(f'{compare_column} = {group2[compare_column].unique()[1]} ({method.capitalize()})')
        
        else:
            fig, axes = plt.subplots(1, len(self.df_list), figsize=(figsize[0] * len(self.df_list), figsize[1]))
            if len(self.df_list) == 1:
                axes = [axes]
                
            for idx, df in enumerate(self.df_list):
                corr_matrix = df[quanti_cols].corr(method=method)
                sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, cmap='coolwarm', ax=axes[idx])
                axes[idx].set_title(f'Dataset {idx + 1} ({method.capitalize()})')

        plt.tight_layout()
        plt.show()

    def plot_pairplot(self, quanti_cols, compare_column=None, palette='Set1', height=3, aspect=1.5):
        """
        Génère un pairplot agrandi pour les colonnes quantitatives, avec une taille horizontale accrue.

        Arguments :
        - quanti_cols : Liste des colonnes quantitatives à comparer.
        - compare_column : Colonne pour colorer les points (par ex. 'target').
        - palette : Palette de couleurs pour la visualisation.
        - height : Hauteur de chaque sous-graphe.
        - aspect : Ratio pour augmenter la taille horizontale.
        """
        for idx, df in enumerate(self.df_list):
            if compare_column:
                sns.pairplot(df, vars=quanti_cols, hue=compare_column, palette=palette, diag_kind='hist', height=height, aspect=aspect)
                plt.suptitle(f"Pairplot - Dataset {idx + 1} avec comparaison sur '{compare_column}'", y=1.02, fontsize=20)
            else:
                sns.pairplot(df, vars=quanti_cols, diag_kind='hist', height=height, aspect=aspect)
                plt.suptitle(f"Pairplot - Dataset {idx + 1} sans comparaison", y=1.02, fontsize=20)

            plt.tight_layout()
            plt.show()

    def plot_pie_charts_comparison(self, categorical_columns, label_df1='DF1', label_df2='DF2'):
        """
        Méthode pour comparer deux DataFrames à l'aide de pie charts pour chaque variable catégorielle.

        Arguments :
        - categorical_columns : Liste des colonnes catégorielles à analyser.
        - label_df1 : Étiquette pour le premier DataFrame (par défaut 'DF1').
        - label_df2 : Étiquette pour le second DataFrame (par défaut 'DF2').

        Affiche deux pie charts côte à côte pour chaque variable catégorielle.
        """
        
        if len(self.df_list) < 2:
            print("Cette méthode nécessite deux DataFrames pour la comparaison.")
            return

        df1 = self.df_list[0]
        df2 = self.df_list[1]
        
        n_cols = 2
        n_rows = math.ceil(len(categorical_columns))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5), squeeze=False)

        for i, col in enumerate(categorical_columns):
            category_data_df1 = df1[col].value_counts()
            axes[i, 0].pie(category_data_df1, labels=category_data_df1.index, autopct='%1.1f%%', startangle=90)
            axes[i, 0].axis('equal')  
            axes[i, 0].set_title(f'{label_df1} - {col}', fontsize=16)

            category_data_df2 = df2[col].value_counts()
            axes[i, 1].pie(category_data_df2, labels=category_data_df2.index, autopct='%1.1f%%', startangle=90)
            axes[i, 1].axis('equal')  
            axes[i, 1].set_title(f'{label_df2} - {col}', fontsize=16)

        plt.tight_layout()
        plt.show()


###########################################################################
#                       Statistical Tests
###########################################################################


class StatisticalTests:
    def __init__(self, df):
        """
        Initialise la classe avec un DataFrame.
        """
        self.df = df

    def test_dependance_quanti_quali2(self, qualitative_vars, quantitative_vars):
        """
        Teste la dépendance statistique entre chaque combinaison de variables qualitatives et quantitatives.

        Parameters:
        - qualitative_vars: Liste de variables qualitatives.
        - quantitative_vars: Liste de variables quantitatives.

        Returns:
        - Un dictionnaire contenant les résultats des tests pour chaque combinaison.
        """
        results = {}
        significant_pairs = []
        
        for qual_var in qualitative_vars:
            for quant_var in quantitative_vars:
                unique_vals = self.df[qual_var].nunique()
                
                if unique_vals == 2:
                    # Test de Mann-Whitney U pour une variable qualitative binaire
                    group1 = self.df[self.df[qual_var] == self.df[qual_var].unique()[0]][quant_var].dropna()
                    group2 = self.df[self.df[qual_var] == self.df[qual_var].unique()[1]][quant_var].dropna()
                    
                    stat, p_value = mannwhitneyu(group1, group2)
                    test_type = "Mann-Whitney U"
                
                elif unique_vals > 2:
                    # Test de Kruskal-Wallis pour plus de deux catégories
                    grouped_data = [group[quant_var].dropna() for name, group in self.df.groupby(qual_var)]
                    stat, p_value = kruskal(*grouped_data)
                    test_type = "Kruskal-Wallis"
                
                # Enregistrer les résultats
                results[f'{qual_var} vs {quant_var}'] = {'Test': test_type, 'Statistic': stat, 'p-value': p_value}
                
                # Afficher les résultats intermédiaires
                print(f"{qual_var} vs {quant_var}:")
                print(f"   Test utilisé: {test_type}")
                print(f"   Statistique: {stat:.4f}")
                print(f"   p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("   Conclusion: Association statistiquement significative\n")
                    significant_pairs.append(f"{qual_var} vs {quant_var}")
                else:
                    print("   Conclusion: Aucune association significative\n")
        
        # Résumé des variables dépendantes
        if significant_pairs:
            print("Résumé - Variables avec dépendance statistiquement significative dans le test de dépendance quanti-quali :")
            print(", ".join(significant_pairs))
        else:
            print("Résumé - Aucune dépendance statistiquement significative trouvée dans le test de dépendance quanti-quali.")
        
        return results

    def test_dependance_quanti_quali(self, qualitative_vars, quantitative_vars):
        results = {}
        significant_pairs = []

        for qual_var in qualitative_vars:
            for quant_var in quantitative_vars:
                unique_vals = self.df[qual_var].nunique()  # Obtenir le nombre de valeurs uniques dans la colonne

                if unique_vals == 2:
                    # Accéder aux valeurs uniques en utilisant iloc[0] et iloc[1]
                    val1, val2 = self.df[qual_var].unique()[:2]  # Assurez-vous d'obtenir les deux premières valeurs uniques
                    group1 = self.df[self.df[qual_var] == val1][quant_var].dropna()
                    group2 = self.df[self.df[qual_var] == val2][quant_var].dropna()

                    stat, p_value = mannwhitneyu(group1, group2)
                    test_type = "Mann-Whitney U"

                elif unique_vals > 2:
                    # Test de Kruskal-Wallis pour plus de deux catégories
                    grouped_data = [group[quant_var].dropna() for _, group in self.df.groupby(qual_var)]
                    stat, p_value = kruskal(*grouped_data)
                    test_type = "Kruskal-Wallis"

                # Enregistrement des résultats
                results[f'{qual_var} vs {quant_var}'] = {'Test': test_type, 'Statistic': stat, 'p-value': p_value}

                # Affichage des résultats intermédiaires
                print(f"{qual_var} vs {quant_var}:")
                print(f"   Test utilisé: {test_type}")
                print(f"   Statistique: {stat:.4f}")
                print(f"   p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("   Conclusion: Association statistiquement significative\n")
                    significant_pairs.append(f"{qual_var} vs {quant_var}")
                else:
                    print("   Conclusion: Aucune association significative\n")

        # Résumé des variables dépendantes
        if significant_pairs:
            print("Résumé - Variables avec dépendance statistiquement significative dans le test quanti-quali :")
            print(", ".join(significant_pairs))
        else:
            print("Résumé - Aucune dépendance statistiquement significative trouvée dans le test quanti-quali.")

        return results

    def apply_tests_anova_ttest(self, quantitative_vars, qualitative_vars):
        """
        Teste la dépendance statistique entre chaque combinaison de variables qualitatives et quantitatives
        avec des tests ANOVA et t-test.

        Parameters:
        - quantitative_vars: Liste de variables quantitatives.
        - qualitative_vars: Liste de variables qualitatives.

        Returns:
        - Un dictionnaire contenant les résultats des tests pour chaque combinaison.
        """
        results = {}
        significant_pairs = []
        
        for qual_var in qualitative_vars:
            for quant_var in quantitative_vars:
                unique_vals = self.df[qual_var].nunique()

                # Vérification que les groupes ne sont pas vides et ont une variance non nulle
                valid_groups = []
                for val in self.df[qual_var].unique():
                    group = self.df[self.df[qual_var] == val][quant_var].dropna()
                    if len(group) > 1 and group.var() > 0:
                        valid_groups.append(group)
                
                if len(valid_groups) < 2:
                    results[f'{qual_var} vs {quant_var}'] = {'Test': "Aucun test effectué", 'Statistic': float('nan'), 'p-value': float('nan')}
                    continue

                if unique_vals == 2 and len(valid_groups) == 2:
                    # Test t de Student pour deux groupes
                    stat, p_value = ttest_ind(valid_groups[0], valid_groups[1], equal_var=False)
                    test_type = "Test t de Student"
                
                elif unique_vals > 2:
                    # ANOVA pour plus de deux groupes
                    stat, p_value = f_oneway(*valid_groups)
                    test_type = "ANOVA"
                
                results[f'{qual_var} vs {quant_var}'] = {'Test': test_type, 'Statistic': stat, 'p-value': p_value}
                
                # Afficher les résultats intermédiaires
                print(f"{qual_var} vs {quant_var}:")
                print(f"   Test utilisé: {test_type}")
                print(f"   Statistique: {stat:.4f}")
                print(f"   p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("   Conclusion: Association statistiquement significative\n")
                    significant_pairs.append(f"{qual_var} vs {quant_var}")
                else:
                    print("   Conclusion: Aucune association significative\n")
        
        # Résumé des variables dépendantes
        if significant_pairs:
            print("Résumé - Variables avec dépendance statistiquement significative dans le test ANOVA/t-test :")
            print(", ".join(significant_pairs))
        else:
            print("Résumé - Aucune dépendance statistiquement significative trouvée dans le test ANOVA/t-test.")
        
        return results

    def test_correlation(self, quantitative_vars, method='pearson'):
        """
        Teste la corrélation entre toutes les combinaisons de variables quantitatives.

        Parameters:
        - quantitative_vars: Liste de variables quantitatives.
        - method: Méthode de corrélation ('pearson' ou 'spearman').

        Returns:
        - Un dictionnaire contenant les résultats des tests de corrélation pour chaque paire de variables.
        """
        results = {}
        significant_pairs = []

        for i, var1 in enumerate(quantitative_vars):
            for var2 in quantitative_vars[i + 1:]:
                if method == 'pearson':
                    corr, p_value = pearsonr(self.df[var1].dropna(), self.df[var2].dropna())
                elif method == 'spearman':
                    corr, p_value = spearmanr(self.df[var1].dropna(), self.df[var2].dropna())
                else:
                    raise ValueError("Méthode de corrélation non supportée. Utilisez 'pearson' ou 'spearman'.")

                results[f'{var1} vs {var2}'] = {
                    'Correlation': corr,
                    'p-value': p_value,
                    'Significantly correlated': p_value < 0.05
                }
                
                # Afficher les résultats intermédiaires
                print(f"{var1} vs {var2}:")
                print(f"   Méthode de corrélation: {method}")
                print(f"   Coefficient de corrélation: {corr:.4f}")
                print(f"   p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print("   Conclusion: Corrélation statistiquement significative\n")
                    significant_pairs.append(f"{var1} vs {var2}")
                else:
                    print("   Conclusion: Aucune corrélation statistiquement significative\n")

        # Résumé des corrélations significatives
        if significant_pairs:
            print("Résumé - Paires de variables avec une corrélation statistiquement significative :")
            print(", ".join(significant_pairs))
        else:
            print("Résumé - Aucune corrélation statistiquement significative trouvée.")

        return results

    def chi2_test(self, target_variable=None, var1=None, var2=None):
        """
        Effectue le test du Chi2 pour :
        - Comparer chaque variable catégorielle dans var1 avec chaque variable dans var2 si les deux sont fournis en tant que listes.
        - Ou, si une variable cible est spécifiée, tester chaque variable dans le DataFrame par rapport à cette variable cible.

        Parameters:
        - target_variable: La variable cible à analyser (mode de comparaison unique).
        - var1, var2: Listes de variables catégorielles pour la comparaison croisée.

        Returns:
        - Un dictionnaire des résultats pour chaque paire de variables ayant une dépendance statistiquement significative.
        """
        significant_results = {}

        # Mode 1 : Comparer chaque variable avec une variable cible unique
        if target_variable:
            for col in self.df.columns:
                if self.df[col].dtype == 'object' and col != target_variable:
                    contingency_table = pd.crosstab(self.df[col], self.df[target_variable])
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    print(f"Variable: {col}, p-value: {p_value:.4f}")
                    
                    if p_value < 0.05:
                        significant_results[f"{col} vs {target_variable}"] = {
                            "chi2_stat": chi2_stat,
                            "p_value": p_value,
                            "significant": True
                        }
                        print(f"   Conclusion: Dépendance statistiquement significative entre '{col}' et '{target_variable}'.\n")
                    else:
                        print(f"   Conclusion: Aucune dépendance significative entre '{col}' et '{target_variable}'.\n")

        # Mode 2 : Comparer toutes les variables de var1 avec celles de var2 (comparaison croisée)
        elif var1 and var2:
            for col1 in var1:
                for col2 in var2:
                    if col1 != col2:  # Éviter la comparaison de la même variable avec elle-même
                        contingency_table = pd.crosstab(self.df[col1], self.df[col2])
                        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                        print(f"{col1} vs {col2}: p-value = {p_value:.4f}")
                        
                        if p_value < 0.05:
                            significant_results[f"{col1} vs {col2}"] = {
                                "chi2_stat": chi2_stat,
                                "p_value": p_value,
                                "significant": True
                            }
                            print(f"   Conclusion: Dépendance statistiquement significative entre '{col1}' et '{col2}'.\n")
                        else:
                            print(f"   Conclusion: Aucune dépendance significative entre '{col1}' et '{col2}'.\n")
        
        # Résumé des variables avec dépendance statistique significative
        if significant_results:
            print("\nRésumé - Paires de variables avec une dépendance statistiquement significative :")
            for pair in significant_results:
                print(f" - {pair}")
        else:
            print("\nRésumé - Aucune dépendance statistiquement significative trouvée entre les paires de variables.")

        return significant_results





###########################################################################
#                       Detection outliers 
###########################################################################


class OutlierDetection:
    def __init__(self, df):
        """
        Initialise la classe avec un DataFrame.
        
        Arguments:
        - df : DataFrame contenant les données.
        """
        self.df = df

    def detect_outliers_zscore(self, numeric_columns, threshold=3):
        """
        Détecte les outliers en utilisant la méthode des Z-scores.
        
        Arguments:
        - numeric_columns : Liste des colonnes numériques à analyser pour la détection des outliers.
        - threshold : Seuil de Z-score pour définir un outlier (par défaut 3).
        
        Retour :
        - df_outliers_zscore : DataFrame contenant uniquement les outliers détectés par Z-score.
        - df_clear_outlier_zscore : DataFrame sans les outliers détectés.
        """
        z_scores = np.abs((self.df[numeric_columns] - self.df[numeric_columns].mean()) / self.df[numeric_columns].std())
        outliers_zscore = (z_scores > threshold).any(axis=1)
        df_outliers_zscore = self.df[outliers_zscore]
        df_clear_outlier_zscore = self.df.drop(df_outliers_zscore.index, errors='ignore')

        print(f"Outliers détectés avec la méthode Z-score (seuil = {threshold}) : {len(df_outliers_zscore)} lignes.")
        return df_outliers_zscore, df_clear_outlier_zscore

    def detect_outliers_iqr(self, numeric_columns):
        """
        Détecte les outliers en utilisant la méthode de l'IQR (Interquartile Range).
        
        Arguments:
        - numeric_columns : Liste des colonnes numériques à analyser.
        
        Retour :
        - df_outliers_iqr : DataFrame contenant uniquement les outliers détectés par IQR.
        - df_clear_outlier_iqr : DataFrame sans les outliers détectés.
        """
        Q1 = self.df[numeric_columns].quantile(0.25)
        Q3 = self.df[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = (self.df[numeric_columns] < (Q1 - 1.5 * IQR)) | (self.df[numeric_columns] > (Q3 + 1.5 * IQR))
        df_outliers_iqr = self.df[outliers_iqr.any(axis=1)]
        df_clear_outlier_iqr = self.df.drop(df_outliers_iqr.index, errors='ignore')

        print(f"Outliers détectés avec la méthode IQR : {len(df_outliers_iqr)} lignes.")
        return df_outliers_iqr, df_clear_outlier_iqr

    @staticmethod
    def compare_dfs(dfs, target_col='embauche'):
        """
        Compare plusieurs DataFrames pour trouver les lignes en commun et les valeurs de la colonne cible spécifiée.
        
        Arguments:
        - dfs : Liste des DataFrames à comparer.
        - target_col : La colonne cible pour compter les valeurs spécifiques (par défaut 'embauche').
        
        Retour :
        - common_count : Nombre total de lignes en commun entre les DataFrames.
        - embauche_common_count : Nombre de lignes avec 'target_col = 1' en commun.
        """
        # Obtenir les indices en commun entre tous les DataFrames
        common_indices = set(dfs[0].index)
        for df in dfs[1:]:
            common_indices = common_indices.intersection(set(df.index))

        # Convertir les indices communs en liste
        common_indices = list(common_indices)

        common_count = len(common_indices)

        # Créer un DataFrame avec les lignes en commun
        if common_count > 0:
            df_common = dfs[0].loc[common_indices]
            embauche_common_count = df_common[df_common[target_col] == 1].shape[0]
        else:
            print("Aucune ligne en commun trouvée.")
            return 0, 0

        print(f"\nNombre total de lignes en commun entre les DataFrames: {common_count}")
        print(f"Nombre de lignes avec '{target_col} = 1' en commun: {embauche_common_count}\n")

        # Calculer et afficher les statistiques pour chaque DataFrame
        for i, df in enumerate(dfs):
            total_rows = df.shape[0]
            embauche_count = df[df[target_col] == 1].shape[0]
            print(f"DataFrame {i + 1}:")
            print(f"  Nombre total de lignes: {total_rows}")
            print(f"  Nombre de '{target_col} = 1': {embauche_count}\n")
        
        return common_count, embauche_common_count



###########################################################################
#                       Feateres engenering 
###########################################################################


class FeatureEngineering:
    def __init__(self, df=None):
        """
        Initialise la classe FeatureEngineering avec un DataFrame optionnel.
        
        Arguments:
        - df : DataFrame contenant les données sur lesquelles effectuer les transformations (par défaut : None).
        """
        self.df = df

    def diviser_sexe_et_statut(self, colonne_source='statut_personnel_et_sexe'):
        """
        Divise la colonne source en deux colonnes distinctes : 'sexe' et 'statut_personnel'.
        Les modalités de 'statut_personnel' seront 'divorcé/séparé/marié/veuf' ou 'célibataire'.
        
        Arguments:
        - colonne_source : Nom de la colonne source à diviser (par défaut : 'statut_personnel_et_sexe').
        
        Retourne:
        - Le DataFrame mis à jour avec les nouvelles colonnes 'sexe' et 'statut_personnel'.
        """
        if self.df is not None and colonne_source in self.df.columns:
            self.df[['sexe', 'statut_personnel_temp']] = self.df[colonne_source].str.split(':', expand=True)
            self.df['statut_personnel'] = self.df['statut_personnel_temp'].apply(
                lambda x: 'divorcé/séparé/marié/veuf' if x in ['divorcé/séparé', 'divorcée/séparée/mariée', 'marié/veuf'] else 'célibataire'
            )
            self.df.drop(columns=['statut_personnel_temp'], inplace=True)
        else:
            print("Erreur : le DataFrame n'est pas défini ou la colonne source est manquante.")
        
        return self.df

    def features_engineering_modele(self):
        """
        Effectue les transformations de feature engineering pour préparer les variables d'entrée du modèle.
        
        Transformations :
        - Création d'une variable combinant les scores de compte chèque et épargne pour une évaluation de la solidité financière.
        - Calcul du montant du crédit par mois.
        - Calcul d'un indice de stabilité financière basé sur plusieurs facteurs financiers, incluant 
        la normalisation du score financier, le montant du crédit par mois et le taux de déchéance.

        - Calcul d'un ratio épargne/crédit.
        - Création d'une variable croisée entre l'ancienneté de la résidence et le type de logement.
        
        Retourne:
        - Le DataFrame mis à jour avec les nouvelles variables générées.
        """
        ###### 1
        # Mapping des scores pour le compte chèque
        cheque_scores = {
            'pas_de_compte_chèque': 0,
            '<0DM': -1,
            '0<=x<200DM': 2,
            '>=200DM/salaires_assignés_pendant_au_moins_1_an': 3
        }
        epargne_scores = {
            'inconnu/pas_de_compte_épargne': 0,
            '<100DM': 1,
            '100<=x<500DM': 2,
            '500<=x<1000DM': 3,
            '>=1000DM': 4
        }

        # Calcul des scores de solidité financière pour le compte chèque et l'épargne
        self.df['cheque_score'] = self.df['statut_du_compte_chèque_existant'].map(cheque_scores)
        self.df['epargne_score'] = self.df['compte_épargne_obligations'].map(epargne_scores)

        # Création de la nouvelle variable 'score_financier'
        self.df['score_financier'] = self.df['cheque_score'] + self.df['epargne_score']

        ######## 2
        # Calcul du montant du crédit par mois
        self.df['montant_du_crédit_par_mois'] = self.df['montant_du_crédit'] / self.df['durée_en_mois']
        
        ######## 3
        # Normalisation et calcul de l'indice de stabilité financière
        self.df['montant_du_crédit_par_mois_normalisé'] = self.df['montant_du_crédit_par_mois'] / self.df['montant_du_crédit_par_mois'].max()
        self.df['score_financier_normalisé'] = self.df['score_financier'] / self.df['score_financier'].max()
        self.df['taux_déchéance_normalisé'] = self.df['taux_déchéance_en_pourcentage_du_revenu_disponible'] / 100

        # Calcul de l'indice de stabilité financière basé sur les variables normalisées
        self.df['indice_stabilité_financière'] = (self.df['score_financier_normalisé'] * 0.5) * (1 / (self.df['montant_du_crédit_par_mois_normalisé'] + 1)) * (1 - (self.df['taux_déchéance_normalisé'] * 0.5))

        # Normalisation de l'indice de stabilité financière
        indice_min = self.df['indice_stabilité_financière'].min()
        indice_max = self.df['indice_stabilité_financière'].max()
        self.df['indice_stabilité_financière_normalisé'] = (self.df['indice_stabilité_financière'] - indice_min) / (indice_max - indice_min)

        ####### 4
        # Calcul du ratio épargne/crédit
        savings_mapping = {
            '<100DM': 50,
            '500<=x<1000DM': 500,
            '>=1000DM': 1000,
            '100<=x<500DM': 100,
            'inconnu/pas_de_compte_épargne': 0
        }
        self.df['Ratio_Épargne_Crédit'] = self.df['compte_épargne_obligations'].map(savings_mapping) / self.df['montant_du_crédit']

        ####### 5
        # Création d'une variable croisée entre l'ancienneté de la résidence et le type de logement
        self.df['croisement_résidence_logement'] = self.df['ancienneté_de_la_résidence_actuelle'].astype(str) + '_' + self.df['logement']

        return self.df



###########################################################################
#                       Discretisation 
###########################################################################


class FeatureDiscretization:
    def __init__(self, df):
        """
        Initialise la classe FeatureDiscretization avec un DataFrame.
        
        Arguments:
        - df : Le DataFrame à traiter.
        """
        self.df = df

    def chi_merge_discretization(self, target, variables, max_intervals=4):
        """
        Discrétisation utilisant la méthode ChiMerge.
        
        Arguments:
        - target : La colonne cible.
        - variables : Liste des variables quantitatives à discrétiser.
        - max_intervals : Nombre maximum d'intervalles.
        
        Retourne:
        - Le DataFrame mis à jour avec les colonnes discrétisées par ChiMerge.
        """
        chi_merge = ChiMergeDiscretizer(target_col=target, max_intervals=max_intervals)
        for var in variables:
            self.df[f'{var}_chimerge'] = chi_merge.fit_transform(self.df[[var]], self.df[target])
        
        print("Discrétisation par ChiMerge terminée.")
        return self.df

    def evaluate_discretization(self, target, variables, test_size=0.3, random_state=42):
        """
        Évalue la performance du modèle sur les variables quantitatives sous forme continue et discrétisée en utilisant la régression logistique.
        
        Arguments:
        - target : Nom de la colonne cible.
        - variables : Liste des variables quantitatives à évaluer.
        - test_size : Proportion des données pour le test.
        - random_state : Graine pour la reproductibilité.
        
        Retourne:
        - scores : Dictionnaire contenant les scores de précision des modèles utilisant les variables continues vs. discrétisées.
        """
        scores = {}

        # Séparation des données en entraînement et test avec les variables quantitatives continues
        X_original = self.df[variables]
        y = self.df[target]
        
        X_train_orig, X_test_orig, y_train, y_test = train_test_split(X_original, y, test_size=test_size, random_state=random_state)
        model_orig = LogisticRegression(random_state=random_state, max_iter=1000)
        model_orig.fit(X_train_orig, y_train)
        y_pred_orig = model_orig.predict(X_test_orig)
        scores['original'] = accuracy_score(y_test, y_pred_orig)
        print(f"Précision du modèle avec variables continues : {scores['original']}")

        # Vérifie que toutes les colonnes discrétisées sont présentes dans le DataFrame pour chaque variable
        discretized_columns = []
        for var in variables:
            col = f'{var}_chimerge'
            if col in self.df.columns:
                discretized_columns.append(col)
            else:
                print(f"Avertissement : Colonne {col} manquante. Assurez-vous que {var} a été discrétisée.")

        # Séparation des données en entraînement et test pour le modèle avec variables discrétisées
        if discretized_columns:
            # Encodage des variables catégorielles si nécessaire
            X_discretized = pd.get_dummies(self.df[discretized_columns], drop_first=True)
            X_train_disc, X_test_disc, y_train, y_test = train_test_split(X_discretized, y, test_size=test_size, random_state=random_state)
            
            model_disc = LogisticRegression(random_state=random_state, max_iter=1000)
            model_disc.fit(X_train_disc, y_train)
            y_pred_disc = model_disc.predict(X_test_disc)
            scores['discretized'] = accuracy_score(y_test, y_pred_disc)
            print(f"Précision du modèle avec variables discrétisées : {scores['discretized']}")
        else:
            print("Aucune colonne discrétisée trouvée pour l'évaluation.")

        return scores


###########################################################################
#                       TRAIN/TEST SPLIT
###########################################################################


from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, df, features, target='target'):
        """
        Initialise la classe DataSplitter.

        Arguments:
        - df : DataFrame contenant les données.
        - features : Liste des colonnes de features à utiliser.
        - target : Nom de la colonne cible pour la stratification (par défaut 'embauche').
        """
        self.df = df
        self.features = features
        self.target = target

    def stratified_train_test_split(self, test_size=0.2, random_state=42):
        """
        Effectue un train_test_split en utilisant la stratification par la classe cible.

        Arguments:
        - test_size : Proportion des données à inclure dans l'ensemble de test (par défaut 0.2).
        - random_state : Seed pour la reproductibilité.

        Retourne :
        - X_train, X_test, y_train, y_test : Les ensembles d'entraînement et de test.
        """
        X = self.df[self.features]
        y = self.df[self.target]

        # Split avec stratification par la colonne cible
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state)

        return X_train, X_test, y_train, y_test


###########################################################################
#                       ENCODAGE 
###########################################################################


class Encodage:
    def __init__(self, X_train, y_train):
        """
        Initialise la classe Encodage avec les données d'entraînement X et y.
        
        Arguments:
        - X_train : DataFrame des variables explicatives d'entraînement.
        - y_train : Série de la variable cible d'entraînement.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.encodings = {}
        self.one_hot_encoder = None

    def target_encoding(self, column):
        """
        Effectue l'encodage target mean (moyenne de la target) sur une colonne spécifique.

        Arguments:
        - column : Nom de la colonne pour laquelle effectuer l'encodage.
        
        Retourne:
        - Applique l'encodage target mean à X_train.
        """
        target_mean_encoding = self.X_train.join(self.y_train).groupby(column)['target'].mean()
        self.encodings[column] = target_mean_encoding
        self.X_train[f'{column}_encoded'] = self.X_train[column].map(target_mean_encoding)
    
    def apply_target_encoding(self, X_test, column):
        """
        Applique l'encodage target mean sur le jeu de test en utilisant les moyennes du jeu d'entraînement.
        
        Arguments:
        - X_test : DataFrame des variables explicatives de test.
        - column : Nom de la colonne pour laquelle appliquer l'encodage.
        
        Retourne:
        - X_test mis à jour avec l'encodage target.
        """
        X_test[f'{column}_encoded'] = X_test[column].map(self.encodings[column]).fillna(self.y_train.mean())
        return X_test

    def count_encoding(self, column):
        """
        Effectue l'encodage count (fréquence d'apparition) pour une colonne spécifique.

        Arguments:
        - column : Nom de la colonne pour laquelle effectuer l'encodage.
        
        Retourne:
        - Applique l'encodage count à X_train.
        """
        count_encoding = self.X_train[column].value_counts()
        self.encodings[column] = count_encoding
        self.X_train[f'{column}_encoded'] = self.X_train[column].map(count_encoding)
    
    def apply_count_encoding(self, X_test, column):
        """
        Applique l'encodage count sur le jeu de test avec remplacement des valeurs inconnues par zéro.
        
        Arguments:
        - X_test : DataFrame des variables explicatives de test.
        - column : Nom de la colonne pour laquelle appliquer l'encodage.
        
        Retourne:
        - X_test mis à jour avec l'encodage count.
        """
        X_test[f'{column}_encoded'] = X_test[column].map(self.encodings[column]).fillna(0)
        return X_test

    def create_binary_encoding(self, column, new_column, true_value='oui'):
        """
        Crée un encodage binaire pour une colonne spécifique en fonction d'une valeur spécifique.

        Arguments:
        - column : Nom de la colonne pour créer l'encodage.
        - new_column : Nom de la nouvelle colonne binaire créée.
        - true_value : Valeur spécifique pour laquelle le binaire sera 0, sinon 1 (par défaut 'oui').
        """
        self.X_train[new_column] = self.X_train[column].apply(lambda x: 0 if x == true_value else 1)

    def manual_encoding(self, column, encoding_map):
        """
        Applique un encodage manuel basé sur un dictionnaire de correspondance.
        
        Arguments:
        - column : Nom de la colonne pour laquelle appliquer l'encodage.
        - encoding_map : Dictionnaire définissant les correspondances pour l'encodage.
        """
        self.X_train[f'{column}_encoded'] = self.X_train[column].map(encoding_map)

    def apply_manual_encoding(self, X_test, column, encoding_map):
        """
        Applique l'encodage manuel sur le jeu de test en utilisant un dictionnaire de correspondance.

        Arguments:
        - X_test : DataFrame des variables explicatives de test.
        - column : Nom de la colonne pour laquelle appliquer l'encodage.
        - encoding_map : Dictionnaire définissant les correspondances pour l'encodage.
        
        Retourne:
        - X_test mis à jour avec l'encodage manuel.
        """
        X_test[f'{column}_encoded'] = X_test[column].map(encoding_map).fillna(-1)
        return X_test
    
    def analyze_categorical_ambiguity(self, categorical_columns_ambigue):
        """
        Analyse les colonnes catégorielles ambiguës et affiche les statistiques par modalité.

        Arguments:
        - categorical_columns_ambigue : Liste des colonnes catégorielles à analyser.
        
        Affiche les pourcentages et effectifs de la target pour chaque modalité.
        """
        for col in categorical_columns_ambigue:
            print(f"Variable : {col}")
            modality_counts = self.df_transformed.groupby(col)['target'].agg(['sum', 'count'])
            modality_counts['percentage_target_1'] = (modality_counts['sum'] / modality_counts['count']) * 100
            modality_counts.rename(columns={'sum': 'effectif_target_1'}, inplace=True)
            modality_counts = modality_counts.sort_values(by='percentage_target_1')
            
            for modality, values in modality_counts.iterrows():
                print(f"  {modality}: {values['percentage_target_1']:.2f}% (effectif target = 1: {values['effectif_target_1']}, total dans la catégorie: {values['count']})")
            print("\n")

    def fit_one_hot_encoding(self, columns):
        """
        Applique l'encodage one-hot sur les colonnes spécifiées du jeu d'entraînement.
        
        Arguments:
        - columns : Liste des colonnes à encoder avec one-hot.
        
        Retourne:
        - X_train mis à jour avec les colonnes encodées en one-hot.
        """
        self.one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
        onehot_encoded_train = self.one_hot_encoder.fit_transform(self.X_train[columns])

        encoded_column_names = [f"{col}_encoded" for col in self.one_hot_encoder.get_feature_names_out(columns)]
        onehot_df_train = pd.DataFrame(onehot_encoded_train, columns=encoded_column_names)

        self.X_train = pd.concat([self.X_train.reset_index(drop=True), onehot_df_train], axis=1)
        self.X_train.drop(columns, axis=1, inplace=True)

        return self.X_train

    def apply_one_hot_encoding(self, X_test, columns):
        """
        Applique l'encodage one-hot sur le jeu de test en utilisant l'encodage appris depuis X_train.

        Arguments:
        - X_test : DataFrame des variables explicatives de test.
        - columns : Liste des colonnes à encoder avec one-hot.
        
        Retourne:
        - X_test mis à jour avec les colonnes encodées en one-hot.
        """
        onehot_encoded_test = self.one_hot_encoder.transform(X_test[columns])
        
        encoded_column_names = [f"{col}_encoded" for col in self.one_hot_encoder.get_feature_names_out(columns)]
        onehot_df_test = pd.DataFrame(onehot_encoded_test, columns=encoded_column_names)
        
        X_test = pd.concat([X_test.reset_index(drop=True), onehot_df_test], axis=1)
        X_test.drop(columns, axis=1, inplace=True)
        
        return X_test



###########################################################################
#                       gestion desequilibre echantillon
###########################################################################

class GestionDesequilibreEchantillon:
    def __init__(self, X_train, y_train):
        """
        Initialise la classe avec les ensembles d'entraînement X_train et y_train.
        """
        self.X_train = X_train
        self.y_train = y_train

    def up_sampling(self):
        """
        Applique le suréchantillonnage pour augmenter la classe minoritaire jusqu'à égaliser la distribution avec la classe majoritaire.
        """
        X = pd.concat([self.X_train, self.y_train], axis=1)
        
        # Sépare les classes majoritaire et minoritaire
        majoritaire = X[self.y_train == self.y_train.value_counts().idxmax()]
        minoritaire = X[self.y_train == self.y_train.value_counts().idxmin()]
        
        # Rééchantillonne la classe minoritaire avec remise
        minoritaire_sur_echantillonnee = resample(minoritaire,
                                                  replace=True,
                                                  n_samples=len(majoritaire),
                                                  random_state=42)
        
        # Combine les ensembles majoritaire et minoritaire suréchantillonné
        echantillon_egalise = pd.concat([majoritaire, minoritaire_sur_echantillonnee])
        
        # Sépare les features (X) et la cible (y)
        self.X_train_up, self.y_train_up = echantillon_egalise.drop(self.y_train.name, axis=1), echantillon_egalise[self.y_train.name]
        return self.X_train_up, self.y_train_up

    def smote(self, target_ratio=0.43):
        """
        Applique SMOTE pour générer des exemples synthétiques de la classe minoritaire
        afin d'atteindre un ratio de défaut (DR) cible spécifique.
        """
        # Calcul du ratio de la classe minoritaire
        n_majoritaire = sum(self.y_train == self.y_train.value_counts().idxmax())
        n_minoritaire = int(n_majoritaire * target_ratio / (1 - target_ratio))
        
        smote = SMOTE(sampling_strategy={self.y_train.value_counts().idxmin(): n_minoritaire}, random_state=42)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train, self.y_train)
        return self.X_train_smote, self.y_train_smote

    def rose(self, target_ratio=0.49):
        """
        Applique ROSE pour créer des exemples artificiels par échantillonnage lissé,
        visant à atteindre un ratio de défaut (DR) cible spécifique.
        """
        # Calcul du ratio de la classe minoritaire
        n_majoritaire = sum(self.y_train == self.y_train.value_counts().idxmax())
        n_minoritaire = int(n_majoritaire * target_ratio / (1 - target_ratio))
        
        rose = RandomOverSampler(sampling_strategy={self.y_train.value_counts().idxmin(): n_minoritaire}, random_state=42)
        self.X_train_rose, self.y_train_rose = rose.fit_resample(self.X_train, self.y_train)
        return self.X_train_rose, self.y_train_rose


# Pour le one hot encoding (plus simple d'utilisation pour cet encodage)
# gestion desequilibre echantillon one hot encoding



class GestionDesequilibreEchantillon2:
    def __init__(self, X_train_onehot_features, y_train):
        """
        Initialise la classe avec les ensembles d'entraînement X_train_onehot_features et y_train.
        """
        self.X_train_onehot_features = X_train_onehot_features
        self.y_train = y_train

    def up_sampling(self):
        """
        Applique le suréchantillonnage pour augmenter la classe minoritaire jusqu'à égaliser la distribution avec la classe majoritaire.
        """
        # Concaténation des données d'entraînement X et y, puis réinitialisation des index
        X = pd.concat([self.X_train_onehot_features.reset_index(drop=True), self.y_train.reset_index(drop=True)], axis=1)
        
        # Sépare les classes majoritaire et minoritaire
        majoritaire = X[X[self.y_train.name] == self.y_train.value_counts().idxmax()]
        minoritaire = X[X[self.y_train.name] == self.y_train.value_counts().idxmin()]
        
        # Rééchantillonne la classe minoritaire avec remise
        minoritaire_sur_echantillonnee = resample(minoritaire,
                                                  replace=True,
                                                  n_samples=len(majoritaire),
                                                  random_state=42)
        
        # Combine les ensembles majoritaire et minoritaire suréchantillonné
        echantillon_egalise = pd.concat([majoritaire, minoritaire_sur_echantillonnee])
        
        # Sépare les features (X) et la cible (y)
        self.X_train_up, self.y_train_up = echantillon_egalise.drop(self.y_train.name, axis=1), echantillon_egalise[self.y_train.name]
        return self.X_train_up, self.y_train_up

    def smote(self, target_ratio=0.43):
        """
        Applique SMOTE pour générer des exemples synthétiques de la classe minoritaire
        afin d'atteindre un ratio de défaut (DR) cible spécifique.
        """
        # Calcul du ratio de la classe minoritaire
        n_majoritaire = sum(self.y_train == self.y_train.value_counts().idxmax())
        n_minoritaire = int(n_majoritaire * target_ratio / (1 - target_ratio))
        
        smote = SMOTE(sampling_strategy={self.y_train.value_counts().idxmin(): n_minoritaire}, random_state=42)
        self.X_train_smote, self.y_train_smote = smote.fit_resample(self.X_train_onehot_features, self.y_train)
        return self.X_train_smote, self.y_train_smote

    def rose(self, target_ratio=0.49):
        """
        Applique ROSE pour créer des exemples artificiels par échantillonnage lissé,
        visant à atteindre un ratio de défaut (DR) cible spécifique.
        """
        # Calcul du ratio de la classe minoritaire
        n_majoritaire = sum(self.y_train == self.y_train.value_counts().idxmax())
        n_minoritaire = int(n_majoritaire * target_ratio / (1 - target_ratio))
        
        rose = RandomOverSampler(sampling_strategy={self.y_train.value_counts().idxmin(): n_minoritaire}, random_state=42)
        self.X_train_rose, self.y_train_rose = rose.fit_resample(self.X_train_onehot_features, self.y_train)
        return self.X_train_rose, self.y_train_rose



###########################################################################
#                       Selection  de Variables
###########################################################################


class SelectionVariable:
    def __init__(self, X_train, y_train, target_name='target'):
        """
        Initialise la classe SelectionVariable avec les données d'entraînement X et y.
        
        Arguments:
        - X_train : DataFrame des variables explicatives d'entraînement.
        - y_train : Série de la variable cible d'entraînement.
        - target_name : Nom de la variable cible (par défaut : 'target').
        """
        self.X_train = X_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.target_name = target_name
        
    def detect_multicollinear_variables(self, quantitative_vars, qualitative_vars, threshold=0.8):
        """
        Détecte les variables multicolinéaires pour les variables quantitatives et qualitatives.

        Arguments:
        - quantitative_vars : Liste des variables quantitatives.
        - qualitative_vars : Liste des variables qualitatives.
        - threshold : Seuil de corrélation pour détecter la multicolinéarité (par défaut = 0.8).

        Retourne:
        - Dictionnaire des paires de variables multicolinéaires détectées.
        """
        multicollinear_pairs = {}

        # Détection de corrélations élevées pour les variables quantitatives
        print("Détection des corrélations élevées pour les variables quantitatives...")
        correlation_matrix = self.X_train[quantitative_vars].corr().abs()
        high_corr_pairs = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                if correlation_matrix.iloc[i, j] > threshold:
                    var1 = correlation_matrix.columns[i]
                    var2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((var1, var2))
                    print(f"Corrélation élevée détectée : {var1} - {var2} avec corrélation = {correlation_matrix.iloc[i, j]:.2f}")

        multicollinear_pairs['quantitative'] = high_corr_pairs

        # Détection des dépendances pour les variables qualitatives
        print("\nDétection des dépendances pour les variables qualitatives...")
        chi2_significant_pairs = []

        for i, var1 in enumerate(qualitative_vars):
            if var1 not in self.X_train.columns:
                print(f"Variable {var1} absente de X_train. Ignorée.")
                continue

            for var2 in qualitative_vars[i + 1:]:
                if var2 not in self.X_train.columns:
                    print(f"Variable {var2} absente de X_train. Ignorée.")
                    continue

                contingency_table = pd.crosstab(self.X_train[var1], self.X_train[var2])
                if contingency_table.size > 0:
                    chi2, p, _, _ = chi2_contingency(contingency_table)
                    if p < 0.05:
                        chi2_significant_pairs.append((var1, var2))
                        print(f"Dépendance significative détectée : {var1} - {var2} avec p-value = {p:.4f}")

        multicollinear_pairs['qualitative'] = chi2_significant_pairs

        return multicollinear_pairs

    def test_dependance_quanti_quali(self, qualitative_vars, quantitative_vars):
        """
        Teste les dépendances entre variables qualitatives et quantitatives en utilisant des tests statistiques.

        Arguments:
        - qualitative_vars : Liste des variables qualitatives.
        - quantitative_vars : Liste des variables quantitatives.

        Retourne:
        - Dictionnaire contenant les résultats des tests et les paires significatives.
        """
        results = {}
        significant_pairs = []

        for qual_var in qualitative_vars:
            for quant_var in quantitative_vars:
                unique_vals = self.X_train[qual_var].nunique()

                if unique_vals == 2:
                    unique_values = self.X_train[qual_var].unique()
                    group1 = self.X_train[self.X_train[qual_var] == unique_values[0]][quant_var].dropna()
                    group2 = self.X_train[self.X_train[qual_var] == unique_values[1]][quant_var].dropna()
                    stat, p_value = mannwhitneyu(group1, group2)
                    test_type = "Mann-Whitney U"
                elif unique_vals > 2:
                    grouped_data = [group[quant_var].dropna() for name, group in self.X_train.groupby(qual_var)]
                    stat, p_value = kruskal(*grouped_data)
                    test_type = "Kruskal-Wallis"
                else:
                    continue

                results[f'{qual_var} vs {quant_var}'] = {'Test': test_type, 'Statistic': stat, 'p-value': p_value}
                if p_value < 0.05:
                    significant_pairs.append(f"{qual_var} vs {quant_var}")

        return results

    def detection_multicolinearite(self, target_variable, quantitative_vars, qualitative_vars):
        """
        Détecte les variables multicolinéaires et conserve celles les plus corrélées avec la cible.

        Arguments:
        - target_variable : Nom de la variable cible binaire.
        - quantitative_vars : Liste des variables quantitatives.
        - qualitative_vars : Liste des variables qualitatives.

        Retourne:
        - Liste des variables conservées après élimination de la multicolinéarité.
        """
        correlation_results = self.test_correlation(quantitative_vars, method='pearson')
        correlated_pairs = [pair for pair, result in correlation_results.items() if result['Significantly correlated']]
        
        dependance_results = self.test_dependance_quanti_quali(qualitative_vars, quantitative_vars)
        significant_pairs = [pair.split(' vs ')[1] for pair, result in dependance_results.items() if result['p-value'] < 0.05]
        
        correlated_vars = set(var for pair in correlated_pairs for var in pair.split(' vs '))
        significant_vars = set(significant_pairs)
        
        multicollinear_vars = list(correlated_vars | significant_vars)

        retained_vars = []
        for var in multicollinear_vars:
            if var in quantitative_vars:
                corr, _ = pearsonr(self.X_train[var], self.y_train)
                retained_vars.append((var, abs(corr)))
            elif var in qualitative_vars:
                chi2_result = self.chi2_test(target_variable=target_variable, var1=[var])
                if chi2_result:
                    chi2_value = chi2_result[f"{var} vs {target_variable}"]["chi2_stat"]
                    retained_vars.append((var, chi2_value))
        
        retained_vars = sorted(retained_vars, key=lambda x: x[1], reverse=True)
        retained_vars = [var[0] for var in retained_vars]

        print("Variables multicolinéaires détectées et conservées (plus haute dépendance avec la cible) :")
        print(retained_vars)
        return retained_vars

    def test_correlation(self, quantitative_vars, method='pearson'):
        """
        Calcule la corrélation entre les variables quantitatives en utilisant le test de Pearson.

        Arguments:
        - quantitative_vars : Liste des variables quantitatives.
        - method : Méthode de corrélation (par défaut 'pearson').

        Retourne:
        - Dictionnaire avec les résultats de la corrélation pour chaque paire de variables.
        """
        results = {}
        for i, var1 in enumerate(quantitative_vars):
            for var2 in quantitative_vars[i + 1:]:
                corr, p_value = pearsonr(self.X_train[var1].dropna(), self.X_train[var2].dropna())
                results[f'{var1} vs {var2}'] = {'Correlation': corr, 'p-value': p_value, 'Significantly correlated': p_value < 0.05}
        return results

    def chi2_test(self, target_variable=None, var1=None):
        """
        Effectue le test du chi2 pour détecter les dépendances entre les variables qualitatives et la cible.

        Arguments:
        - target_variable : Nom de la variable cible.
        - var1 : Liste des variables qualitatives.

        Retourne:
        - Dictionnaire contenant les résultats significatifs du test du chi2.
        """
        significant_results = {}
        if target_variable:
            for col in self.X_train.columns:
                if col != target_variable and pd.api.types.is_categorical_dtype(self.X_train[col]):
                    contingency_table = pd.crosstab(self.X_train[col], self.y_train)
                    chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                    if p_value < 0.05:
                        significant_results[f"{col} vs {target_variable}"] = {"chi2_stat": chi2_stat, "p_value": p_value}
        return significant_results

    def stepwise_selection(self, model=LogisticRegression(max_iter=1000, solver='liblinear')):
        """
        Sélection pas à pas des variables en utilisant un modèle de régression logistique.

        Arguments:
        - model : Modèle de régression logistique pour la sélection des variables.

        Retourne:
        - Liste des variables sélectionnées.
        """
        initial_features = self.X_train.columns.tolist()
        selected_features = []
        remaining_features = initial_features[:]

        while remaining_features:
            best_candidate = None
            best_score = -np.inf
            
            for candidate in remaining_features:
                model.fit(self.X_train[selected_features + [candidate]], self.y_train)
                score = model.score(self.X_train[selected_features + [candidate]], self.y_train)
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            if best_candidate:
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
            else:
                break

        print(f'Variables sélectionnées avec la sélection pas à pas : {selected_features}')
        return selected_features

    def stepwise_selection2(self, model=LogisticRegression(max_iter=1000, solver='liblinear')):
        """
        Sélection pas à pas avec BIC pour évaluer la pertinence des variables.

        Arguments:
        - model : Modèle de régression logistique pour la sélection des variables.

        Retourne:
        - Liste des variables sélectionnées.
        """
        initial_features = self.X_train.columns.tolist()
        selected_features = []
        remaining_features = initial_features[:]

        def calculate_bic(n, log_likelihood, n_params):
            """Calcule le BIC en fonction de la taille de l'échantillon, la log-vraisemblance et le nombre de paramètres."""
            return n_params * np.log(n) - 2 * log_likelihood

        while remaining_features:
            best_candidate = None
            lowest_bic = np.inf
            
            for candidate in remaining_features:
                model.fit(self.X_train[selected_features + [candidate]], self.y_train)
                y_pred_proba = model.predict_proba(self.X_train[selected_features + [candidate]])
                log_likelihood = -log_loss(self.y_train, y_pred_proba, normalize=False)
                n = len(self.y_train)
                n_params = len(selected_features) + 1

                bic = calculate_bic(n, log_likelihood, n_params)

                if bic < lowest_bic:
                    lowest_bic = bic
                    best_candidate = candidate

            if best_candidate:
                selected_features.append(best_candidate)
                remaining_features.remove(best_candidate)
            else:
                break

        print(f'Variables sélectionnées avec la sélection pas à pas (BIC) : {selected_features}')
        return selected_features
    

    def rfecv_selection(self, model=LogisticRegression(max_iter=1000), step=1, cv=5, scoring='f1'):
        """
        Sélection des variables par RFECV avec validation croisée.

        Arguments:
        - model : Modèle de régression pour la sélection.
        - step : Nombre de variables éliminées à chaque itération.
        - cv : Nombre de plis pour la validation croisée.
        - scoring : Métrique de scoring pour RFECV.

        Retourne:
        - Liste des variables sélectionnées.
        """
        rfecv = RFECV(estimator=model, step=step, cv=StratifiedKFold(cv), scoring=scoring)
        rfecv.fit(self.X_train, self.y_train)

        selected_features = self.X_train.columns[rfecv.support_]
        print(f'Variables sélectionnées avec RFECV : {selected_features.tolist()}')
        print(f'Nombre optimal de variables : {rfecv.n_features_}')

        plt.figure(figsize=(10, 6))
        plt.xlabel("Nombre de variables sélectionnées")
        plt.ylabel(f"Score {scoring}")
        plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'])
        plt.title(f"RFECV - Sélection des variables ({scoring})")
        plt.show()
        return selected_features

    def extra_trees_select(self, model=ExtraTreesClassifier(n_estimators=100, random_state=42), threshold='mean'):
        """
        Sélection des variables avec Extra Trees et SelectFromModel.

        Arguments:
        - model : Modèle Extra Trees pour l'évaluation de l'importance.
        - threshold : Seuil d'importance des variables (par défaut 'mean').

        Retourne:
        - Liste des variables sélectionnées.
        """
        model.fit(self.X_train, self.y_train)
        selector = SelectFromModel(model, threshold=threshold, prefit=True)
        selected_features = self.X_train.columns[selector.get_support()].tolist()

        print(f'Variables sélectionnées avec Extra Trees et SelectFromModel : {selected_features}')
        return selected_features

    def detect_multicollinearity_with_importance(self, model=RandomForestClassifier(n_estimators=100, random_state=42), importance_threshold=0.01, correlation_threshold=0.8):
        """
        Détecte la multicolinéarité en utilisant un modèle pour évaluer l'importance des variables.

        Arguments:
        - model : Modèle pour évaluer l'importance des variables.
        - importance_threshold : Seuil d'importance.
        - correlation_threshold : Seuil de corrélation pour la multicolinéarité.

        Retourne:
        - Liste des variables sélectionnées après élimination de la multicolinéarité.
        """
        X_encoded = pd.get_dummies(self.X_train, drop_first=True)
        
        model.fit(X_encoded, self.y_train)
        feature_importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
        
        selected_features = feature_importances[feature_importances > importance_threshold].index.tolist()
        X_selected = X_encoded[selected_features]
        
        print(f'Variables avec importance supérieure au seuil ({importance_threshold}):')
        print(selected_features)
        
        corr_matrix = X_selected.corr().abs()
        to_remove = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    colname1 = corr_matrix.columns[i]
                    colname2 = corr_matrix.columns[j]
                    
                    if feature_importances[colname1] >= feature_importances[colname2]:
                        to_remove.add(colname2)
                    else:
                        to_remove.add(colname1)

        final_features = [col for col in selected_features if col not in to_remove]
        
        print(f'Variables après élimination des multicolinéarités : {final_features}')
        return final_features

    def random_forest_with_random_feature(self, model=RandomForestClassifier(n_estimators=100, random_state=42)):
        """
        Sélection des variables avec Random Forest, en ajoutant une variable aléatoire comme référence.

        Arguments:
        - model : Modèle Random Forest pour la sélection des variables.

        Retourne:
        - Liste des variables sélectionnées avec une importance supérieure à celle de la variable aléatoire.
        """
        X_train_with_random = self.X_train.copy()
        X_train_with_random['random_feature'] = np.random.rand(len(self.X_train))

        model.fit(X_train_with_random, self.y_train)
        feature_importances = pd.Series(model.feature_importances_, index=X_train_with_random.columns)

        selected_features = feature_importances[feature_importances > feature_importances['random_feature']].index.tolist()
        print(f'Variables sélectionnées avec importance supérieure à la variable aléatoire : {selected_features}')

        feature_importances.sort_values().plot(kind='barh', figsize=(10, 8), title="Importance des variables avec Random Forest")
        plt.show()

        return selected_features



###########################################################################
#                       ModelTraining
###########################################################################


class ModelTraining:
    def __init__(self, X_train, X_test, y_train, y_test, quantitative_vars=None):
        self.X_train = X_train.copy()
        self.X_test = X_test.copy()
        self.y_train = y_train
        self.y_test = y_test
        self.quantitative_vars = quantitative_vars

    def prepare_data(self, scale=False, features=None):
        """
        Applique un scaling optionnel sur les variables quantitatives dans X_train et X_test.
        Permet également de sélectionner des colonnes spécifiques pour X_train et X_test.
        
        Arguments:
        - scale: Booléen indiquant si le scaling des variables quantitatives doit être appliqué.
        - features: Liste des noms de colonnes à garder dans X_train et X_test. Si None, toutes les colonnes sont gardées.
        
        Retourne X_train et X_test après le scaling et la sélection des colonnes si applicable.
        """
        # Filtrer les colonnes si des features spécifiques sont fournies
        if features:
            print("Sélection de colonnes spécifique activée.")
            self.X_train = self.X_train[features]
            self.X_test = self.X_test[features]
        
        scaler = MinMaxScaler()
        
        if scale and self.quantitative_vars:
            print("Scaling activé pour les variables quantitatives.")
            self.X_train[self.quantitative_vars] = scaler.fit_transform(self.X_train[self.quantitative_vars])
            self.X_test[self.quantitative_vars] = scaler.transform(self.X_test[self.quantitative_vars])
        else:
            print("Scaling désactivé.")
        
        return self.X_train, self.X_test

    def train_model(self, model):
        """
        Entraîne un modèle sur les données d'entraînement.
        Arguments:
        - model: Le modèle à entraîner.
        Retourne le modèle entraîné.
        """
        model.fit(self.X_train, self.y_train)
        return model

    def grid_search(self, model, param_grid, cv=5, scoring='f1'):
        """
        Effectue une recherche par grille (Grid Search) sur un modèle donné et retourne le meilleur modèle.
        Arguments:
        - model: Le modèle à optimiser.
        - param_grid: Grille de paramètres pour le Grid Search.
        - cv: Nombre de plis pour la validation croisée.
        - scoring: La métrique à optimiser (par défaut 'f1').
        Retourne le meilleur modèle trouvé.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=StratifiedKFold(cv), scoring=scoring, verbose=2)
        grid_search.fit(self.X_train, self.y_train)
        print(f"Meilleur modèle : {grid_search.best_params_}")
        return grid_search.best_estimator_

    def bayesian_optimization(self, model, param_space, cv=5, scoring='f1', n_iter=20):
        """
        Effectue une optimisation bayésienne sur un modèle donné et retourne le meilleur modèle.
        Arguments:
        - model: Le modèle à optimiser.
        - param_space: Espace des hyperparamètres pour l'optimisation bayésienne.
        - cv: Nombre de plis pour la validation croisée.
        - scoring: La métrique à optimiser (par défaut 'f1').
        - n_iter: Nombre d'itérations pour l'optimisation.
        Retourne le meilleur modèle trouvé.
        """
        bayes_search = BayesSearchCV(
            estimator=model,
            search_spaces=param_space,
            n_iter=n_iter,
            cv=StratifiedKFold(cv),
            scoring=scoring,
            verbose=2,
            random_state=42
        )
        bayes_search.fit(self.X_train, self.y_train)
        print(f"Meilleur modèle avec optimisation bayésienne : {bayes_search.best_params_}")
        return bayes_search.best_estimator_

    def find_optimal_threshold(self, model):
        predicted_probabilities = model.predict_proba(self.X_test)[:, 1]  # Probabilités de la classe positive
        fpr, tpr, thresholds = roc_curve(self.y_test, predicted_probabilities)
        roc_auc = auc(fpr, tpr)

        # Calcul du seuil optimal
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        print("AUC :", roc_auc.round(2))
        print("Seuil optimal de classification :", optimal_threshold.round(4))
        return optimal_threshold

    def plot_proba_distribution(self, model, dataset='train'):
        """
        Plot the probability distribution for y=0 and y=1 classes.
        """
        if dataset == 'train':
            proba = model.predict_proba(self.X_train)[:, 1]
            y = self.y_train
            title = "Distribution de predict_proba pour y_train = 0 et y_train = 1"
        else:
            proba = model.predict_proba(self.X_test)[:, 1]
            y = self.y_test
            title = "Distribution de predict_proba pour y_test = 0 et y_test = 1"
        
        proba_df = pd.DataFrame({f'y_{dataset}': y, f'y_{dataset}_proba': proba})
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=proba_df, x=f'y_{dataset}_proba', hue=f'y_{dataset}', element='step', stat='density', common_norm=False)
        plt.title(title)
        plt.xlabel("Probabilité prédite de la classe positive")
        plt.ylabel("Densité")
        plt.legend(title='Classe réelle', labels=[f'y_{dataset} = 0', f'y_{dataset} = 1'])
        plt.show()

    def apply_custom_threshold(self, model, threshold=0.5):
        """
        Apply a custom threshold on the probabilities to determine predicted classes.
        """
        y_test_proba = model.predict_proba(self.X_test)[:, 1]
        y_test_pred_custom_threshold = (y_test_proba >= threshold).astype(int)
        return y_test_pred_custom_threshold

    def evaluate_with_custom_threshold(self, model, threshold=0.5):
        """
        Evaluate the model using a custom threshold and print the evaluation metrics.
        """
        y_test_pred_custom_threshold = self.apply_custom_threshold(model, threshold)
        y_test_proba = model.predict_proba(self.X_test)[:, 1]

        # Calcul des métriques
        f1 = f1_score(self.y_test, y_test_pred_custom_threshold)
        accuracy = accuracy_score(self.y_test, y_test_pred_custom_threshold)
        precision = precision_score(self.y_test, y_test_pred_custom_threshold)
        recall = recall_score(self.y_test, y_test_pred_custom_threshold)
        auc_score = roc_auc_score(self.y_test, y_test_proba)

        # Affichage des métriques
        print(f"Évaluation avec un seuil de {threshold}:")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print("\nClassification report:")
        print(classification_report(self.y_test, y_test_pred_custom_threshold))

        # Affichage de la matrice de confusion
        print("Matrice de confusion:")
        conf_matrix = confusion_matrix(self.y_test, y_test_pred_custom_threshold)
        print(conf_matrix)



###########################################################################
#                       Evaluation et Features importances
###########################################################################


class VariableImportanceEvaluation:
    def __init__(self, X_train, y_train, X_test=None, y_test=None):
        """
        Initialise la classe VariableImportanceEvaluation avec les ensembles d'entraînement et de test.
        
        Arguments:
        - X_train : DataFrame des variables explicatives d'entraînement.
        - y_train : Série de la variable cible d'entraînement.
        - X_test : (Optionnel) DataFrame des variables explicatives de test.
        - y_test : (Optionnel) Série de la variable cible de test.
        """
        self.X_train = X_train.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True) if X_test is not None else None
        self.y_test = y_test.reset_index(drop=True) if y_test is not None else None

    def feature_importance(self, model):
        """
        Affiche l'importance des features pour un modèle donné.
        
        Arguments:
        - model : Modèle entraîné avec lequel calculer l'importance des features.
        """
        model.fit(self.X_train, self.y_train)
        
        if hasattr(model, 'feature_importances_'):
            feature_importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            feature_importances = abs(model.coef_).flatten()
        else:
            raise AttributeError(f"Le modèle {type(model).__name__} n'a pas d'attribut feature_importances_ ou coef_.")

        features_importance_df = pd.DataFrame({'Feature': self.X_train.columns, 'Importance': feature_importances})
        features_importance_df = features_importance_df.sort_values(by='Importance', ascending=False)
        
        print(features_importance_df)

        plt.figure(figsize=(10, 6))
        palette = sns.color_palette("viridis", len(features_importance_df))
        sns.barplot(x='Importance', y='Feature', data=features_importance_df, palette=palette)
        
        plt.title(f'Importance des Features - {type(model).__name__}')
        plt.show()

    def shap_explanation(self, model, variables=None, figsize=(10, 6), interaction_variable=None):
        """
        Génère des explications SHAP pour visualiser l'importance des features.

        Arguments:
        - model : Modèle entraîné pour générer les valeurs SHAP.
        - variables : Liste des features à inclure dans les explications SHAP (par défaut : toutes les features).
        - figsize : Taille des graphiques SHAP.
        - interaction_variable : (Optionnel) Nom de la feature pour un graphique d'interaction SHAP.
        """
        if variables is None:
            variables = self.X_train.columns

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_train[variables])

        plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, self.X_train[variables], plot_type="dot", feature_names=variables)

        plt.figure(figsize=figsize)
        shap.summary_plot(shap_values, self.X_train[variables], plot_type="bar", feature_names=variables)

        if interaction_variable:
            plt.figure(figsize=figsize)
            shap.dependence_plot(interaction_variable, shap_values, self.X_train[variables], interaction_index=interaction_variable, feature_names=variables)

    def evaluate_model(self, model):
        """
        Évalue les performances du modèle sur l'ensemble d'entraînement et de test (si disponible).
        
        Arguments:
        - model : Modèle entraîné à évaluer.
        """
        y_pred_train = model.predict(self.X_train)
        print("Évaluation sur l'ensemble d'entraînement :")
        print(classification_report(self.y_train, y_pred_train))
        
        conf_matrix_train = confusion_matrix(self.y_train, y_pred_train)
        plt.figure(figsize=(6, 4))
        sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title("Matrice de confusion - Entraînement")
        plt.xlabel("Prédiction")
        plt.ylabel("Vraie Valeur")
        plt.show()

        if self.X_test is not None and self.y_test is not None:
            y_pred_test = model.predict(self.X_test)
            print("Évaluation sur l'ensemble de test :")
            print(classification_report(self.y_test, y_pred_test))
            
            conf_matrix_test = confusion_matrix(self.y_test, y_pred_test)
            plt.figure(figsize=(6, 4))
            sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title("Matrice de confusion - Test")
            plt.xlabel("Prédiction")
            plt.ylabel("Vraie Valeur")
            plt.show()

            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_prob)
                roc_auc = roc_auc_score(self.y_test, y_prob)

                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='b')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel("Taux de faux positifs (FPR)")
                plt.ylabel("Taux de vrais positifs (TPR)")
                plt.title("Courbe ROC - Test")
                plt.legend(loc="lower right")
                plt.show()
                
                print(f"AUC Score: {roc_auc:.2f}")
            else:
                print("Le modèle ne supporte pas predict_proba, AUC non calculé.")

    def grid_search_and_evaluate(self, model, params, model_name=""):
        """
        Effectue une recherche de grille pour optimiser le modèle, puis évalue le modèle optimal.
        
        Arguments:
        - model : Modèle à optimiser.
        - params : Grille de paramètres pour la recherche.
        - model_name : Nom du modèle pour l'affichage des résultats.
        """
        grid_search = GridSearchCV(estimator=model, param_grid=params, scoring='f1', cv=5, verbose=1)
        grid_search.fit(self.X_train, self.y_train)
        
        best_model = grid_search.best_estimator_
        self.evaluate_model(best_model)
        
        print(f"Meilleurs paramètres pour {model_name}: {grid_search.best_params_}")

    def prepare_data(self, quantitative_vars, categorical_vars, scale=False, model_name=None):
        """
        Prépare les données pour l'entraînement en appliquant une mise à l'échelle et en convertissant les variables catégorielles.
        
        Arguments:
        - quantitative_vars : Liste des variables quantitatives à scaler.
        - categorical_vars : Liste des variables catégorielles à encoder.
        - scale : Indique si les variables quantitatives doivent être mises à l'échelle.
        - model_name : Nom du modèle pour décider si le scaling est nécessaire.
        """
        if scale and model_name not in ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost']:
            scaler = MinMaxScaler()
            self.X_train[quantitative_vars] = scaler.fit_transform(self.X_train[quantitative_vars])
            self.X_test[quantitative_vars] = scaler.transform(self.X_test[quantitative_vars])

        for col in categorical_vars:
            self.X_train[col] = self.X_train[col].astype('int')
            self.X_test[col] = self.X_test[col].astype('int')
            

    def convert_period_columns(self, period_columns):
        """
        Convertit les colonnes de périodes en entiers pour faciliter l'analyse.
        
        Arguments:
        - period_columns : Liste des colonnes à convertir en format entier.
        """
        for col in period_columns:
            self.X_train[col] = self.X_train[col].astype(str).astype(int)
            self.X_test[col] = self.X_test[col].astype(str).astype(int)

    def print_test_metrics(self, model):
        """
        Affiche les métriques de performance pour le modèle sur l'ensemble de test (si disponible).
        
        Arguments:
        - model : Modèle à évaluer.
        """
        if self.X_test is not None and self.y_test is not None:
            y_pred = model.predict(self.X_test)
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(self.X_test)[:, 1]
                auc = roc_auc_score(self.y_test, y_prob)
            else:
                auc = None
            
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F1 Score: {f1:.2f}")
            if auc is not None:
                print(f"AUC: {auc:.2f}")
            else:
                print("AUC: Non calculé, le modèle ne supporte pas predict_proba.")
        else:
            print("Ensemble de test non fourni.")


###########################################################################
#                     Fonction pour le notebook 
###########################################################################

###########################################################################
#                     Comparaison encodage 
###########################################################################


def compare_encodings_models(encodings, models, quantitative_vars):
    """
    Compare les performances de différents modèles pour différents types d'encodage.

    :param encodings: Dictionnaire contenant les jeux de données et les features pour chaque encodage.
    :param models: Dictionnaire contenant les modèles à tester.
    :param quantitative_vars: Liste des variables quantitatives.
    :return: DataFrame contenant les performances des modèles pour chaque type d'encodage.
    """
    results = []

    for encoding_name, data in encodings.items():
        X_train_enc = data["X_train"]
        X_test_enc = data["X_test"]
        y_train_enc = data["y_train"]
        y_test_enc = data["y_test"]
        feature_set = data["features"]

        for model_name, model in models.items():
            scale = True if isinstance(model, LogisticRegression) else False

            # Préparation des données avec ou sans scaling selon le modèle
            model_training = ModelTraining(X_train_enc, X_test_enc, y_train_enc, y_test_enc, quantitative_vars=quantitative_vars)
            X_train_scaled, X_test_scaled = model_training.prepare_data(scale=scale, features=feature_set)

            # Entraîner le modèle
            trained_model = model_training.train_model(model)

            # Prédictions et calcul des métriques de performance
            y_pred = trained_model.predict(X_test_scaled)
            y_proba = trained_model.predict_proba(X_test_scaled)[:, 1] if hasattr(trained_model, "predict_proba") else None

            accuracy = accuracy_score(y_test_enc, y_pred)
            precision = precision_score(y_test_enc, y_pred, zero_division=0)
            recall = recall_score(y_test_enc, y_pred, zero_division=0)
            f1 = f1_score(y_test_enc, y_pred, zero_division=0)
            auc = roc_auc_score(y_test_enc, y_proba) if y_proba is not None else None

            results.append({
                "Encoding": encoding_name,
                "Model": model_name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "AUC": auc
            })

    results_df = pd.DataFrame(results)
    return results_df


def logistic_regression_stats(X_train, y_train, feature_names, quantitative_features=None):
    """
    Effectue une régression logistique avec statsmodels pour afficher les coefficients,
    leurs significativités et leurs p-values.

    :param X_train: Données d'entraînement.
    :param y_train: Cible d'entraînement.
    :param feature_names: Liste des noms des features.
    :param quantitative_features: Liste des noms des colonnes quantitatives à normaliser.
    :return: Résumé de la régression logistique avec les p-values et les coefficients.
    """
    X_train, y_train = X_train.align(y_train, join='inner', axis=0)
    
    if quantitative_features:
        scaler = MinMaxScaler()
        X_train[quantitative_features] = scaler.fit_transform(X_train[quantitative_features])

    X_train = sm.add_constant(X_train)
    
    logit_model = sm.Logit(y_train, X_train)
    result = logit_model.fit(disp=0)
    
    summary = result.summary2().tables[1]
    summary.index = ['Intercept'] + feature_names
    return summary


def evaluate_rf_logistic(encodings, quantitative_vars):
    """
    Entraîne et évalue un modèle RandomForest avec un encodage LabelEncoding_target_count_encoding
    et un modèle LogisticRegression avec un encodage OneHotEncoding_labelEncoding.

    :param encodings: Dictionnaire contenant les jeux de données et les features pour chaque type d'encodage.
    :param quantitative_vars: Liste des variables quantitatives.
    """
    rf_data = encodings["LabelEncoding_target_count_encoding"]
    X_train_rf, X_test_rf = rf_data["X_train"], rf_data["X_test"]
    y_train_rf, y_test_rf = rf_data["y_train"], rf_data["y_test"]
    features_rf = rf_data["features"]

    model_training_rf = ModelTraining(X_train_rf, X_test_rf, y_train_rf, y_test_rf, quantitative_vars=quantitative_vars)
    X_train_scaled_rf, X_test_scaled_rf = model_training_rf.prepare_data(scale=False, features=features_rf)
    
    rf_model = RandomForestClassifier(n_estimators=150, criterion='gini', random_state=42)
    trained_rf_model = model_training_rf.train_model(rf_model)

    print("Evaluation pour RandomForest avec LabelEncoding_target_count_encoding :")
    evaluator_rf = VariableImportanceEvaluation(X_train_scaled_rf, y_train_rf, X_test_scaled_rf, y_test_rf)
    evaluator_rf.feature_importance(trained_rf_model)
    evaluator_rf.evaluate_model(trained_rf_model)
    
    lr_data = encodings["OneHotEncoding_labelEncoding"]
    X_train_lr, X_test_lr = lr_data["X_train"], lr_data["X_test"]
    y_train_lr, y_test_lr = lr_data["y_train"], lr_data["y_test"]
    features_lr = lr_data["features"]

    model_training_lr = ModelTraining(X_train_lr, X_test_lr, y_train_lr, y_test_lr, quantitative_vars=quantitative_vars)
    X_train_scaled_lr, X_test_scaled_lr = model_training_lr.prepare_data(scale=True, features=features_lr)
    
    lr_model = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    trained_lr_model = model_training_lr.train_model(lr_model)

    print("\nEvaluation pour LogisticRegression avec OneHotEncoding_labelEncoding :")
    evaluator_lr = VariableImportanceEvaluation(X_train_scaled_lr, y_train_lr, X_test_scaled_lr, y_test_lr)
    evaluator_lr.feature_importance(trained_lr_model)
    evaluator_lr.evaluate_model(trained_lr_model)


def evaluate_feature_selection_with_encodings(encodings, quantitative_vars):
    """
    Exécute différentes méthodes de sélection de variables pour chaque type d'encodage et évalue
    les performances d'un modèle Random Forest pour chaque méthode et encodage.

    :param encodings: Dictionnaire contenant les jeux de données et les features pour chaque type d'encodage.
    :param quantitative_vars: Liste des variables quantitatives.
    :return: Deux DataFrames - un avec les performances du modèle et un autre avec les variables sélectionnées.
    """
    results = []
    selected_features = []

    for encoding_name, data in encodings.items():
        X_train = data["X_train"]
        X_test = data["X_test"]
        y_train = data["y_train"]
        y_test = data["y_test"]

        selector = SelectionVariable(X_train, y_train)
        
        rf_with_random_features = selector.random_forest_with_random_feature()
        X_train_rf = X_train[rf_with_random_features]
        X_test_rf = X_test[rf_with_random_features]
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_rf, y_train)
        y_pred_rf = rf_model.predict(X_test_rf)
        y_proba_rf = rf_model.predict_proba(X_test_rf)[:, 1]
        
        results.append({
            'Encoding': encoding_name,
            'Method': 'Random Forest avec feature aléatoire',
            'Accuracy': accuracy_score(y_test, y_pred_rf),
            'Precision': precision_score(y_test, y_pred_rf, zero_division=0),
            'Recall': recall_score(y_test, y_pred_rf, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred_rf, zero_division=0),
            'AUC': roc_auc_score(y_test, y_proba_rf)
        })
        
        selected_features.append({
            'Encoding': encoding_name,
            'Method': 'Random Forest avec feature aléatoire',
            'Selected Features': rf_with_random_features
        })

        extra_trees_features = selector.extra_trees_select()
        X_train_extra_trees = X_train[extra_trees_features]
        X_test_extra_trees = X_test[extra_trees_features]
        
        rf_model.fit(X_train_extra_trees, y_train)
        y_pred_extra_trees = rf_model.predict(X_test_extra_trees)
        y_proba_extra_trees = rf_model.predict_proba(X_test_extra_trees)[:, 1]
        
        results.append({
            'Encoding': encoding_name,
            'Method': 'Extra Trees avec SelectFromModel',
            'Accuracy': accuracy_score(y_test, y_pred_extra_trees),
            'Precision': precision_score(y_test, y_pred_extra_trees, zero_division=0),
            'Recall': recall_score(y_test, y_pred_extra_trees, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred_extra_trees, zero_division=0),
            'AUC': roc_auc_score(y_test, y_proba_extra_trees)
        })

        selected_features.append({
            'Encoding': encoding_name,
            'Method': 'Extra Trees avec SelectFromModel',
            'Selected Features': extra_trees_features
        })

        rfecv_features = selector.rfecv_selection(model=RandomForestClassifier(n_estimators=100, random_state=42))
        X_train_rfecv = X_train[rfecv_features]
        X_test_rfecv = X_test[rfecv_features]
        
        rf_model.fit(X_train_rfecv, y_train)
        y_pred_rfecv = rf_model.predict(X_test_rfecv)
        y_proba_rfecv = rf_model.predict_proba(X_test_rfecv)[:, 1]
        
        results.append({
            'Encoding': encoding_name,
            'Method': 'RFECV',
            'Accuracy': accuracy_score(y_test, y_pred_rfecv),
            'Precision': precision_score(y_test, y_pred_rfecv, zero_division=0),
            'Recall': recall_score(y_test, y_pred_rfecv, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred_rfecv, zero_division=0),
            'AUC': roc_auc_score(y_test, y_proba_rfecv)
        })

        selected_features.append({
            'Encoding': encoding_name,
            'Method': 'RFECV',
            'Selected Features': rfecv_features.tolist()
        })

    results_df = pd.DataFrame(results)
    selected_features_df = pd.DataFrame(selected_features)
    
    return results_df, selected_features_df


def evaluate_logistic_selection(X_train, X_test, y_train, y_test, features):
    """
    Évalue les méthodes de sélection de variables pour la régression logistique
    sur les données d'entraînement et de test.

    :param X_train: Données d'entraînement.
    :param X_test: Données de test.
    :param y_train: Cible d'entraînement.
    :param y_test: Cible de test.
    :param features: Liste des features à utiliser.
    :return: DataFrame des résultats et dictionnaire des features sélectionnées.
    """
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train[features]), columns=features)
    X_test = pd.DataFrame(scaler.transform(X_test[features]), columns=features)

    selector = SelectionVariable(X_train, y_train)
    results = []
    selected_features_dict = {}

    stepwise_features = selector.stepwise_selection()
    selected_features_dict['Stepwise Selection'] = stepwise_features
    X_train_stepwise = X_train[stepwise_features]
    X_test_stepwise = X_test[stepwise_features]
    
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train_stepwise, y_train)
    y_pred = model.predict(X_test_stepwise)
    y_proba = model.predict_proba(X_test_stepwise)[:, 1]
    
    results.append({
        'Method': 'Stepwise Selection',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_proba)
    })

    rfecv_features = selector.rfecv_selection2()
    selected_features_dict['RFECV'] = rfecv_features.tolist()
    X_train_rfecv = X_train[rfecv_features]
    X_test_rfecv = X_test[rfecv_features]
    
    model.fit(X_train_rfecv, y_train)
    y_pred = model.predict(X_test_rfecv)
    y_proba = model.predict_proba(X_test_rfecv)[:, 1]
    
    results.append({
        'Method': 'RFECV',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_test, y_proba)
    })

    results_df = pd.DataFrame(results)
    return results_df, selected_features_dict


def detect_multicollinear_variables(X_train, y_train, quantitative_vars, qualitative_vars, threshold=0.8):
    """
    Détecte les variables multicolinéaires entre les variables quantitatives et qualitatives.

    Parameters:
    - X_train: DataFrame des variables explicatives.
    - y_train: Série de la variable cible.
    - quantitative_vars: Liste de variables quantitatives.
    - qualitative_vars: Liste de variables qualitatives.
    - threshold: Seuil de corrélation pour détecter la multicolinéarité pour les variables quantitatives (par défaut = 0.8).

    Returns:
    - Dictionnaire des paires de variables multicolinéaires détectées.
    """
    multicollinear_pairs = {}

    # 1. Détecter les corrélations élevées pour les variables quantitatives
    print("Détection des corrélations élevées pour les variables quantitatives...")
    correlation_matrix = X_train[quantitative_vars].corr().abs()
    high_corr_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if correlation_matrix.iloc[i, j] > threshold:
                var1 = correlation_matrix.columns[i]
                var2 = correlation_matrix.columns[j]
                high_corr_pairs.append((var1, var2))
                print(f"Corrélation élevée détectée : {var1} - {var2} avec corrélation = {correlation_matrix.iloc[i, j]:.2f}")

    multicollinear_pairs['quantitative'] = high_corr_pairs

    # 2. Détecter les dépendances pour les variables qualitatives
    print("\nDétection des dépendances pour les variables qualitatives...")
    chi2_significant_pairs = []

    for i, var1 in enumerate(qualitative_vars):
        if var1 not in X_train.columns:
            print(f"Variable {var1} absente de X_train. Ignorée.")
            continue

        for var2 in qualitative_vars[i + 1:]:
            if var2 not in X_train.columns:
                print(f"Variable {var2} absente de X_train. Ignorée.")
                continue

            # Vérification que les colonnes sont bien des séries 1D
            col1 = X_train[var1]
            col2 = X_train[var2]

            if not (isinstance(col1, pd.Series) and isinstance(col2, pd.Series)):
                print(f"Les colonnes {var1} ou {var2} ne sont pas 1D et seront ignorées.")
                continue

            # Calcul du test du Chi2 pour vérifier la dépendance
            contingency_table = pd.crosstab(col1, col2)
            if contingency_table.size > 0:
                chi2, p, _, _ = chi2_contingency(contingency_table)
                if p < 0.05:
                    chi2_significant_pairs.append((var1, var2))
                    print(f"Dépendance significative détectée : {var1} - {var2} avec p-value = {p:.4f}")

    multicollinear_pairs['qualitative'] = chi2_significant_pairs

    return multicollinear_pairs


def evaluate_sampling_methods(model, X_train_datasets, y_train_datasets, X_test, y_test, method_names, scaling=False, quantitative_features=None):
    """
    Évalue un modèle sur plusieurs ensembles d'entraînement et retourne les métriques de performance.

    Parameters:
    - model: Modèle à entraîner et évaluer.
    - X_train_datasets: Liste des DataFrames des différents ensembles d'entraînement.
    - y_train_datasets: Liste des Series correspondantes des labels des différents ensembles d'entraînement.
    - X_test: DataFrame des données de test.
    - y_test: Series des labels de test.
    - method_names: Liste des noms des méthodes d'échantillonnage (doit correspondre à l'ordre de X_train_datasets et y_train_datasets).
    - scaling: Booléen, optionnel (par défaut=False). Si True, applique la normalisation sur les variables quantitatives spécifiées.
    - quantitative_features: Liste des noms des colonnes à normaliser si `scaling=True`.

    Returns:
    - DataFrame des métriques de performance (Accuracy, Precision, Recall, F1 Score, AUC) pour chaque méthode.
    """
    results = []

    # Vérifier si le scaling est activé et que des features quantitatives sont fournies
    if scaling and quantitative_features is not None:
        scaler = MinMaxScaler()

    for i, (X_train, y_train, method) in enumerate(zip(X_train_datasets, y_train_datasets, method_names)):
        if scaling and quantitative_features is not None:
            # Appliquer le scaling sur les variables quantitatives spécifiées
            X_train_scaled = X_train.copy()
            X_test_scaled = X_test.copy()

            X_train_scaled[quantitative_features] = scaler.fit_transform(X_train[quantitative_features])
            X_test_scaled[quantitative_features] = scaler.transform(X_test[quantitative_features])
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test

        # Entraîner le modèle sur l'ensemble d'entraînement
        model.fit(X_train_scaled, y_train)
        
        # Prédire sur l'ensemble de test
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        # Calculer les métriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None

        # Ajouter les résultats à la liste
        results.append({
            "Sampling Method": method,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc
        })

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)
    return results_df


###########################################################################
#                       EVALUATION MODELE
###########################################################################


# Fonction pour effectuer une recherche de grille et évaluer les modèles
def evaluate_models(X_train_select, y_train, X_test_select, y_test,
                    X_train_onehot_select, y_train_onehot, X_test_onehot_select, y_test_onehot,
                    quantitative_features=None):
    """
    Évalue plusieurs modèles avec une recherche de grille, en appliquant un scaling MinMax aux features
    quantitatives spécifiées pour les modèles non basés sur des arbres.

    Paramètres :
    - X_train_select, y_train, X_test_select, y_test : Jeux de données d'entraînement et de test originaux
    - X_train_onehot_select, y_train_onehot, X_test_onehot_select, y_test_onehot : Jeux de données avec encodage one-hot pour la régression logistique
    - quantitative_features : Liste des features quantitatives à scaler
    """
    for model_name, (model, params) in models.items():
        print(f"Entraînement du modèle : {model_name}")
        
        # Sélectionner le dataset approprié pour la régression logistique (nécessite un encodage one-hot)
        if model_name == 'LogisticRegression':
            X_train = X_train_onehot_select
            y_train = y_train_onehot 
            X_test = X_test_onehot_select
            y_test = y_test_onehot
        else:
            X_train = X_train_select
            X_test = X_test_select

        # Appliquer le scaling MinMax aux features quantitatives pour les modèles non basés sur des arbres
        if model_name not in ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees', 'AdaBoost'] and quantitative_features is not None:
            scaler = MinMaxScaler()
            X_train = X_train.copy()
            X_test = X_test.copy()
            X_train[quantitative_features] = scaler.fit_transform(X_train[quantitative_features])
            X_test[quantitative_features] = scaler.transform(X_test[quantitative_features])

        # Obtenir la métrique d'optimisation pour le modèle actuel
        metric = model_metrics[model_name]
        
        # Effectuer la recherche de grille
        grid_search = GridSearchCV(model, params, scoring=metric, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Obtenir le meilleur modèle et évaluer ses performances
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None
        
        # Calculer les métriques de performance
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Stocker les résultats
        performance_metrics.append({
            "Model": model_name,
            "Optimized For": metric,
            "Best Params": grid_search.best_params_,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "AUC": auc
        })
        print(f"Meilleurs paramètres pour {model_name}: {grid_search.best_params_}")


# Fonction pour effectuer une validation croisée et évaluer les modèles
def evaluate_models_with_cv(X_train, y_train, quantitative_features=None):
    """
    Évalue plusieurs modèles avec une validation croisée à 5 plis, en appliquant un scaling MinMax aux features
    quantitatives spécifiées pour les modèles non basés sur des arbres.

    Paramètres :
    - X_train, y_train : Données d'entraînement et labels
    - quantitative_features : Liste des features quantitatives à scaler
    """
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, (model, params) in models.items():
        print(f"Entraînement du modèle : {model_name}")
        
        # Appliquer le scaling MinMax aux features quantitatives pour les modèles non basés sur des arbres
        if model_name not in ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost', 'ExtraTrees', 'AdaBoost'] and quantitative_features is not None:
            scaler = MinMaxScaler()
            X_train_scaled = X_train.copy()
            X_train_scaled[quantitative_features] = scaler.fit_transform(X_train[quantitative_features])
        else:
            X_train_scaled = X_train
        
        # Obtenir la métrique pour l'optimisation
        metric = model_metrics[model_name]
        
        # Initialiser les listes pour stocker les résultats de chaque pli
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        auc_scores = []
        
        # Effectuer la validation croisée
        for train_index, val_index in kf.split(X_train_scaled, y_train):
            X_train_fold, X_val_fold = X_train_scaled.iloc[train_index], X_train_scaled.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            
            # Entraîner le modèle
            model.fit(X_train_fold, y_train_fold)
            
            # Prédictions sur le set de validation
            y_pred = model.predict(X_val_fold)
            y_pred_proba = model.predict_proba(X_val_fold)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Calculer les métriques pour chaque pli
            accuracy_scores.append(accuracy_score(y_val_fold, y_pred))
            precision_scores.append(precision_score(y_val_fold, y_pred, zero_division=0))
            recall_scores.append(recall_score(y_val_fold, y_pred))
            f1_scores.append(f1_score(y_val_fold, y_pred))
            if y_pred_proba is not None:
                auc_scores.append(roc_auc_score(y_val_fold, y_pred_proba))
        
        # Stocker les métriques moyennes sur tous les plis
        performance_metrics.append({
            "Model": model_name,
            "Accuracy": np.mean(accuracy_scores),
            "Precision": np.mean(precision_scores),
            "Recall": np.mean(recall_scores),
            "F1 Score": np.mean(f1_scores),
            "AUC": np.mean(auc_scores) if auc_scores else None
        })
        print(f"Validation croisée complétée pour {model_name}")


###########################################################################
#                       Enregistrement et chargement du modele 
###########################################################################


class ModelPersistence:
    @staticmethod
    def save_model(model, filename="best_catboost_model.joblib"):
        """
        Enregistre le modèle CatBoost en utilisant joblib.
        
        Arguments:
        - model : Le modèle CatBoost à sauvegarder.
        - filename : Nom du fichier pour sauvegarder le modèle (par défaut "best_catboost_model.joblib").
        """
        joblib.dump(model, filename)
        print(f"Modèle sauvegardé sous {filename}")

    @staticmethod
    def load_model(filename="best_catboost_model.joblib"):
        """
        Charge le modèle CatBoost enregistré avec joblib.
        
        Arguments:
        - filename : Nom du fichier contenant le modèle sauvegardé (par défaut "best_catboost_model.joblib").
        
        Retourne:
        - Le modèle CatBoost chargé.
        """
        model = joblib.load(filename)
        print(f"Modèle chargé depuis {filename}")
        return model





