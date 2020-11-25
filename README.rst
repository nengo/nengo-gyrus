*********************
Nengo Python Template
*********************

This is a template repository for new projects.
When you create a new repository from this template,
follow these instructions:

#. Make sure the ``.gitignore`` file is appropriate for the new repository
   (defaults to a fairly general Python ``.gitignore``).

#. Update ``.nengobones.yml`` with information for the new repository. If there are
   files that the new repository does not require, those sections can be removed to
   avoid generating the file.

   #. ``project_name``: The full name of the project. We tend to name our projects
      in CamelCase, but it can include spaces if you wish.

   #. ``pkg_name``: The package name, as used in ``import package_name``.

   #. ``repo_name``: The repository name, in the form ``organization/repository-name``.
      On Github, the URL should be ``https://github.com/organization/repository-name``.
      The ``organization`` defaults to ``nengo``, but could also be ``abr``.
      The ``repository-name`` is typically the ``package_name`` but with underscores
      replaced with hyphens.

   #. ``copyright_start``: Update this to the current year.

   #. ``setup_py``: This generates ``setup.py``.

      - Requirements have been left in all the ``_req`` sections to demonstrate them.
        If any are not required, they can be removed (e.g. ``scipy``).
      - The ``Framework :: Nengo`` classifier should only be used for projects that are
        part of the Nengo framework.

   #. ``docs_conf_py``:

      - To use Google Analytics for the docs, update the ``tagmanager_id``, otherwise
        remove it.

#. Run ``bones-generate`` to use NengoBones to generate the new files, and add them
   to the repository.

#. Rename the ``package_name`` directory to reflect the package name.

#. Update ``package_name/version.py``, putting in the ``project_name``, ``package_name``
   and initial version.

#. Update ``CHANGES.rst``. The demonstration entry shows how to properly reference a
   PR. If your package is a Nengo backend, we typically note the Nengo versions it is
   compatible with at the top, to make it easier for users using an older Nengo version
   to find compatible backend versions.

#. Replace the text in this ``README.rst`` with text for the new repository.

#. Add all package code to the renamed ``package_name`` directory.

#. Add Jupyter Notebook or plain Python examples to ``docs/examples``.

#. Add Sphinx documentation to ``docs``. For an example of multi-page documentation,
   see `Nengo <https://github.com/nengo/nengo>`_. For an example of single-page
   documentation, see `pytest-plt <https://github.com/nengo/pytest-plt>`_ (note the use
   of the ``one_page`` option in ``.nengobones.yml``).
