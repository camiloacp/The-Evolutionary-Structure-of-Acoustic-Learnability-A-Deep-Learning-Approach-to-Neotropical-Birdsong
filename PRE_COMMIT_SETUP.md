# Configuración de Pre-commit con PyPI Privado

## Problema Resuelto

Este proyecto tenía problemas con pre-commit debido a la configuración global de pip que apuntaba a un servidor PyPI privado corporativo (`pypi.artifacts.furycloud.io`).

## Solución Aplicada

1. **Limpieza de configuración duplicada**: Se eliminó el archivo `pre-commit-config.yaml` duplicado.

2. **Corrección de configuración**: Se removió la configuración incorrecta `pip_options` del archivo `.pre-commit-config.yaml`.

3. **Limpieza de cache**: Se limpió el cache de pre-commit: `pre-commit clean`

4. **Resolución de conflictos de Git hooks**: Se removieron configuraciones conflictivas de `core.hooksPath`.

5. **Instalación con PyPI público**: Se configuró el entorno para usar PyPI público durante la instalación:
   ```bash
   export PIP_INDEX_URL=https://pypi.org/simple
   export PIP_TRUSTED_HOST=pypi.org
   pre-commit install --install-hooks
   ```

## Para Futuros Problemas

Si vuelves a tener problemas con pre-commit, ejecuta:

```bash
# Limpiar cache
pre-commit clean

# Configurar entorno y reinstalar
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_TRUSTED_HOST=pypi.org
pre-commit install --install-hooks
```

## Estado Actual

✅ Pre-commit instalado correctamente
✅ Hooks funcionando (trailing-whitespace, end-of-file-fixer, check-yaml)
✅ Commits funcionando sin errores
