"""Initial migration

Revision ID: 9da54d17067b
Revises: 
Create Date: 2024-08-25 19:21:26.977242

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9da54d17067b'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('historical_properties',
    sa.Column('id', sa.String(), nullable=False),
    sa.Column('price', sa.Integer(), nullable=False),
    sa.Column('date_of_transaction', sa.Date(), nullable=False),
    sa.Column('postal_code', sa.String(), nullable=False),
    sa.Column('property_type', sa.Enum('DETACHED', 'SEMI_DETACHED', 'TERRACED', 'FLAT', 'OTHER', name='propertytype'), nullable=False),
    sa.Column('property_age', sa.Enum('NEW', 'OLD', name='propertyage'), nullable=False),
    sa.Column('duration', sa.Enum('FREEHOLD', 'LEASEHOLD', name='propertyduration'), nullable=False),
    sa.Column('paon', sa.String(), nullable=True),
    sa.Column('saon', sa.String(), nullable=True),
    sa.Column('street', sa.String(), nullable=True),
    sa.Column('locality', sa.String(), nullable=True),
    sa.Column('town_city', sa.String(), nullable=True),
    sa.Column('district', sa.String(), nullable=True),
    sa.Column('ppd_category_type', sa.Enum('STANDARD_PRICE_PAID', 'ADDITIONAL_PRICE_PAID', name='ppdcategorytype'), nullable=False),
    sa.Column('record_status', sa.Enum('ADDITION', 'DELETION', name='recordstatus'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_historical_properties_id'), 'historical_properties', ['id'], unique=False)
    op.create_table('listing_properties',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('property_url', sa.String(), nullable=False),
    sa.Column('title', sa.String(), nullable=False),
    sa.Column('address', sa.String(), nullable=False),
    sa.Column('price', sa.String(), nullable=False),
    sa.Column('pricing_qualifier', sa.String(), nullable=True),
    sa.Column('listing_time', sa.String(), nullable=True),
    sa.Column('property_type', sa.String(), nullable=False),
    sa.Column('bedrooms', sa.String(), nullable=True),
    sa.Column('bathrooms', sa.String(), nullable=True),
    sa.Column('epc_rating', sa.String(), nullable=True),
    sa.Column('size', sa.String(), nullable=True),
    sa.Column('features', sa.JSON(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('property_url')
    )
    op.create_index(op.f('ix_listing_properties_id'), 'listing_properties', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_listing_properties_id'), table_name='listing_properties')
    op.drop_table('listing_properties')
    op.drop_index(op.f('ix_historical_properties_id'), table_name='historical_properties')
    op.drop_table('historical_properties')
    # ### end Alembic commands ###
