"""Nombres de pila frecuentes en Ecuador y el mundo hispano (listado propio, SUPUESTO 4).

No son datos personales: son nombres comunes usados como disparador de la regla de nombres.
Se excluyen nombres que también son palabras frecuentes (Luz, Paz, Rosa, Mercedes, Cruz,
Dolores, Sol, Amparo...) para no borrar texto útil.
"""

FIRST_NAMES: frozenset[str] = frozenset(
    name.lower()
    for name in """
    Adrián Alejandra Alejandro Alexandra Alexis Alicia Ana Andrea Andrés Ángel Ángela Antonio
    Ariel Belén Bryan Camila Carla Carlos Carmen Catalina Cecilia Christian Cristian Cristina
    Daniel Daniela David Diana Diego Domenica Doménica Eduardo Elena Elizabeth Emilia Emilio
    Enrique Erika Esteban Estefanía Fabián Fernanda Fernando Francisco Gabriel Gabriela Galo
    Gloria Guillermo Gustavo Héctor Hugo Isabel Ivonne Jaime Javier Jennifer Jéssica Jessica
    Joel Jorge José Josué Juan Julia Julio Karina Karla Kevin Laura Leonardo Leticia Lorena
    Lucía Luis Manuel Marco Marcos Margarita María Mariana Mario Marta Martha Martín Mateo
    Mauricio Mayra Miguel Mónica Nancy Natalia Nelson Nicolás Norma Olga Óscar Oswaldo Pablo
    Patricia Patricio Paúl Paula Pedro Rafael Ramiro Raúl Ricardo Roberto Rodrigo Rubén Samuel
    Santiago Sara Sebastián Sergio Silvia Sofía Susana Teresa Tomás Valentina Valeria Verónica
    Vicente Victoria Víctor Viviana Wilson Ximena Xavier Yolanda
    """.split()
)
