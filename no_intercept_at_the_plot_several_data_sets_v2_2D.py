"""Apply an independently supplied internal-field coefficient.
The zero-intercept scan is removed. This is neither a low-field cutoff nor
subtraction of a chi*H contribution from magnetisation.
"""
from diagnostics_cli import main
if __name__=='__main__': main(field_correction=True)
