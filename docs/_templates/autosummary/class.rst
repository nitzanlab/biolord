{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

{% block attributes %}
{% if attributes %}
{{ 'Attributes table' | underline(line='-') }}

.. autosummary::
{% for item in attributes %}
{%- if item not in inherited_members and item not in ['training'] %}
    ~{{ fullname }}.{{ item }}
{%- endif %}
{%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
{{ 'Methods table' | underline(line='-') }}

.. autosummary::
{% for item in methods %}
    {%- if item != '__init__' and item not in inherited_members %}
    ~{{ fullname }}.{{ item }}
    {%- endif -%}
{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes_documentation %}
{% if attributes %}
{{ 'Attributes' | underline(line='-') }}
{% for item in attributes %}
{%- if item not in inherited_members and item not in ['training'] %}

{{ item | escape | underline(line='^') }}

.. autoattribute:: {{ [objname, item] | join(".") }}
{%- endif %}
{%- endfor %}

{% endif %}
{% endblock %}

{% block methods_documentation %}
{% if methods %}
{{ 'Methods' | underline(line='-') }}
{% for item in methods %}
{%- if item != '__init__' and item not in inherited_members %}

{{ item | escape | underline(line='^') }}

.. automethod:: {{ [objname, item] | join(".") }}
{%- endif -%}
{%- endfor %}

{% endif %}
{% endblock %}
