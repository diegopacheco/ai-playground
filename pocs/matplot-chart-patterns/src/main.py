import matplotlib.pyplot as plt
import numpy as np

# Parabolic Arc
x = np.linspace(-10, 10, 400)
y = x**2
plt.plot(x, y, label='Parabolic Arc')
plt.title('Tesla Stock Price (2010-2020)')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Inverse Parabolic Arc
x = np.linspace(-10, 10, 400)
y = -x**2
plt.plot(x, y, label='Inverse Parabolic Arc')
plt.title('Bitcoin Price (2017-2018)')
plt.xlabel('Month')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Hyperbola
x = np.linspace(-10, 10, 400)
y = 1/x
plt.plot(x, y, label='Hyperbola')
plt.title('Amazon Stock Price (2001-2010)')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Cup and Handle
x = np.linspace(-10, 10, 400)
y = np.sin(x) + 2
plt.plot(x, y, label='Cup and Handle')
plt.title('Gold Price (2011-2013)')
plt.xlabel('Month')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Inverse Cup and Handle
x = np.linspace(-10, 10, 400)
y = -np.sin(x) + 2
plt.plot(x, y, label='Inverse Cup and Handle')
plt.title('Silver Price (2011-2013)')
plt.xlabel('Month')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Rounded Bottom
x = np.linspace(-10, 10, 400)
y = np.arctan(x)
plt.plot(x, y, label='Rounded Bottom')
plt.title('Coca-Cola Stock Price (2009-2011)')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Rounded Top
x = np.linspace(-10, 10, 400)
y = -np.arctan(x)
plt.plot(x, y, label='Rounded Top')
plt.title('Apple Stock Price (2012-2014)')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Saucer
x = np.linspace(-10, 10, 400)
y = np.sin(x) + 1
plt.plot(x, y, label='Saucer')
plt.title('Microsoft Stock Price (2009-2011)')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Inverse Saucer
x = np.linspace(-10, 10, 400)
y = -np.sin(x) + 1
plt.plot(x, y, label='Inverse Saucer')
plt.title('IBM Stock Price (2012-2014)')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.legend()
plt.show()

# Fanolini
x = np.linspace(-10, 10, 400)
y = x**3
plt.plot(x, y, label='Fanolini')
plt.title('Netflix Stock Price (2010-2012)')
plt.xlabel('Year')
plt.ylabel('Price ($)')
plt.legend()
plt.show()