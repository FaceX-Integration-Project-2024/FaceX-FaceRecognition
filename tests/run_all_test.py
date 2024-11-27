import unittest


## MARCHE PAS ENCORE

if __name__ == "__main__":
    # decouvrir et charger tous les tests du répertoire tests et ses sous dossiers
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir="./tests", pattern="test_*.py")
    print(test_suite)

    # exécuter tous les tests trouvés
    test_runner = unittest.TextTestRunner()
    test_runner.run(test_suite)
