! ------------------------------------------------------------------------------
! Subroutines to compute the distance matrix
! and the bond list of a set of atoms
! ------------------------------------------------------------------------------

subroutine distance_matrix(coordinates, radii, N, distances, nbonds, bonds) bind(C, name="distance_matrix")
    !! @brief computes the distance matrix and the bond list
    !!
    !! @param[in]        N              Number of atoms
    !! @param[in]        coordinates    Cartesian coordinates of the atoms
    !! @param[in]        radii          Radii of the atoms
    !! @param[in,out]    distances      Distance matrix
    !! @param[in,out]    nbonds         Number of bonds
    !! @param[in,out]    bonds          Bond list
    !! 
    !! @author: Rony Letona
    use iso_c_binding
    implicit none

    integer :: dim = 3
    integer(c_int), intent(in), value :: N
    real(c_double), intent(in) :: radii(N)
    real(c_double), intent(in) :: coordinates(N,dim)
    real(c_double), intent(inout) :: distances(N,N)
    integer(c_int), intent(inout) :: nbonds
    integer(c_int), intent(inout) :: bonds(N*N,2)

    integer :: i, j, q
    real(c_double) :: min_bond

    ! Output distance matrix
    distances = 0.0d0

    ! Bond list index
    nbonds = 1

    ! Loop over all atoms
    do i = 1, N
        ! Loop over some atoms
        do j = i+1, N
            ! Loop over all dimensions
            do q = 1, dim
                distances(i,j) = distances(i,j) + (coordinates(i,q) - coordinates(j,q))**2
            end do
            ! Square root of the sum of squares
            distances(i,j) = sqrt(distances(i,j))
            ! Construct symmetric matrix
            distances(j,i) = distances(i,j)
            ! Compute the minimal bonding distance
            min_bond = (radii(i) + radii(j)) * 1.1
            ! Check if the atoms are bonded
            if (distances(i,j) < min_bond) then
                ! Build bonding key and store the bond
                bonds(nbonds,:) = (/i-1, j-1/)
                nbonds = nbonds + 1
            end if
        end do
    end do
end subroutine distance_matrix

subroutine find_bonds_spacepartition(coords, radii, N, box_length, tolerance, nbonds, bonds) bind(C, name="find_bonds")
    !! @brief computes the bond list using a space partition algorithm
    !!
    !! @param[in]        N              Number of atoms
    !! @param[in]        radii          Radii of the atoms
    !! @param[in]        coords         Cartesian coordinates of the atoms
    !! @param[in]        box_length     Length of the simulation box
    !! @param[in]        tolerance      Bonding tolerance
    !! @param[in,out]    nbonds         Number of bonds
    !! @param[in,out]    bonds          Bond list
    !! 
    !! @author: Rony Letona
    use iso_c_binding
    implicit none

    integer(c_int), value :: N
    real(c_double), intent(in) :: radii(N)
    real(c_double), intent(in) :: coords(N,3)
    real(c_double), value :: box_length
    real(c_double), value :: tolerance
    integer(c_int), intent(inout) :: nbonds
    integer(c_int), intent(inout) :: bonds(4*N,2)

    ! Local variables
    real(8) :: xyz_min(3), xyz_max(3), box_len(3)
    integer :: nx, ny, nz, n_cells, d, i, j, k, l, max_atoms_cell
    integer, allocatable :: atom_cell(:), cell_counts(:)
    integer, allocatable :: cell_atoms(:,:), atom_bonded(:,:)
    real(8) :: min_bond, d2
    integer :: this_cell, neigh_cell
    integer :: idx_a, idx_b, cx, cy, cz, nx_, ny_, nz_
    integer :: nbonds_max, bond_idx
    integer :: dx, dy, dz
    logical :: is_bonded, a_saved, b_saved
    integer, parameter :: neigh_offsets(3,7) = reshape([ &
        0, 0, 1,  0, 1, 0,  0, 1, 1, &
        1, 0, 0,  1, 0, 1,  1, 1, 0, &
        1, 1, 1 ], [3, 7])
    
    ! Bounds and cell size
    xyz_min = minval(coords, dim=1)
    xyz_max = maxval(coords, dim=1)
    box_len = xyz_max - xyz_min
    nx = int(box_len(1) / box_length) + 1
    ny = int(box_len(2) / box_length) + 1
    nz = int(box_len(3) / box_length) + 1
    if (nx <= 0 .or. ny <= 0 .or. nz <= 0) then
        nbonds = 0
        return
    endif
    n_cells = nx * ny * nz

    ! Allocate temporaries
    allocate(atom_cell(N))            ! Which cell corresponds to each atom
    allocate(cell_counts(n_cells))    ! How many atoms does this cell have
    allocate(atom_bonded(N,8))        ! Which atoms are bonded to each atom
    ! 8 atoms bonded to a single atom seems overkill, but still ... better be safe

    cell_counts = 0
    atom_bonded = 0

    ! Assign atoms to cells
    do i = 1, N
        cx = int( (coords(i,1) - xyz_min(1)) / box_length ) + 1
        cy = int( (coords(i,2) - xyz_min(2)) / box_length ) + 1
        cz = int( (coords(i,3) - xyz_min(3)) / box_length ) + 1
        ! Clamp
        cx = max(1, min(cx, nx))
        cy = max(1, min(cy, ny))
        cz = max(1, min(cz, nz))
        this_cell = cx + nx*(cy-1) + nx*ny*(cz-1)
        atom_cell(i) = this_cell
        cell_counts(this_cell) = cell_counts(this_cell) + 1
    enddo

    ! Find what is the maximum number of atoms in a cell
    max_atoms_cell = maxval(cell_counts)

    allocate(cell_atoms(n_cells, max_atoms_cell)) ! Which atoms are in each cell
    cell_atoms = 0

    ! Fill cell_atoms (global atom indices per cell)
    do i = 1, N
        this_cell = atom_cell(i)
        do j = 1, cell_counts(this_cell)
            if (cell_atoms(this_cell,j) == 0) then
                cell_atoms(this_cell,j) = i
                exit
            endif
        enddo
    enddo

    ! Estimate max bonds and allocate
    nbonds_max = nint(4.0d0 * N)  ! Rough heuristic: avg 4 neighbors
    bond_idx = 0

    ! Loop over all cells
    do cz = 1, nz
        do cy = 1, ny
            do cx = 1, nx
                ! Get this cell's address
                this_cell = cx + nx*(cy-1) + nx*ny*(cz-1)

                ! Loop over all pairs in this cell
                do idx_a = 1, cell_counts(this_cell) - 1
                    i = cell_atoms(this_cell, idx_a)
                    do idx_b = idx_a + 1, cell_counts(this_cell)
                        j = cell_atoms(this_cell, idx_b)
                        ! Compute the squared distance
                        d2 = 0
                        do d = 1, 3
                            d2 = d2 + (coords(i,d) - coords(j,d))**2
                        enddo
                        ! Compute the minimal bonding distance ...
                        min_bond = (radii(i) + radii(j)) * tolerance
                        min_bond = min_bond * min_bond ! ... squared
                        if (d2 < min_bond * min_bond) then
                            ! This is a bond

                            ! Check if the bond has already been found
                            is_bonded = .false.
                            do l = 1, 8
                                if (atom_bonded(i,l) == j) then
                                    is_bonded = .true.
                                    exit
                                endif
                                if (atom_bonded(j,l) == i) then
                                    is_bonded = .true.
                                    exit
                                endif
                            enddo
                            ! If the bond has been found, skip
                            if (is_bonded) cycle

                            ! if not, add it (for BOTH atoms)
                            a_saved = .false.
                            b_saved = .false.
                            do l = 1, 8
                                if (atom_bonded(i,l) == 0 .and. .not. a_saved) then
                                    atom_bonded(i,l) = j
                                    a_saved = .true.
                                endif
                                if (atom_bonded(j,l) == 0 .and. .not. b_saved) then
                                    atom_bonded(j,l) = i
                                    b_saved = .true.
                                endif
                                if (a_saved .and. b_saved) exit
                            enddo
                            bond_idx = bond_idx + 1
                            bonds(bond_idx, 1) = min(i,j)
                            bonds(bond_idx, 2) = max(i,j)
                        endif
                    enddo
                enddo

                ! Neighbor cells (full 2x2x2 stencil: self + 7 offsets)
                do k = 1, 7
                    dx = neigh_offsets(1,k)
                    dy = neigh_offsets(2,k)
                    dz = neigh_offsets(3,k)
                    nx_ = cx + dx
                    ny_ = cy + dy
                    nz_ = cz + dz
                    if (nx_ < 1 .or. nx_ > nx .or. &
                        ny_ < 1 .or. ny_ > ny .or. &
                        nz_ < 1 .or. nz_ > nz) cycle
                    neigh_cell = nx_ + nx*(ny_-1) + nx*ny*(nz_-1)
                    if (neigh_cell == this_cell) cycle  ! Already handled

                    ! All cross pairs
                    do idx_a = 1, cell_counts(this_cell)
                        i = cell_atoms(this_cell, idx_a)
                        do idx_b = 1, cell_counts(neigh_cell)
                            j = cell_atoms(neigh_cell, idx_b)
                            ! Compute the squared distance
                            d2 = 0
                            do d = 1, 3
                                d2 = d2 + (coords(i,d) - coords(j,d))**2
                            enddo
                            ! Compute the minimal bonding distance ...
                            min_bond = (radii(i) + radii(j)) * tolerance
                            min_bond = min_bond * min_bond ! ... squared
                            if (d2 < min_bond) then
                                ! This is a bond

                                ! Check if the bond has already been found
                                is_bonded = .false.
                                do l = 1, 8
                                    if (atom_bonded(i,l) == j) then
                                        is_bonded = .true.
                                        exit
                                    endif
                                    if (atom_bonded(j,l) == i) then
                                        is_bonded = .true.
                                        exit
                                    endif
                                enddo
                                ! If the bond has been found, skip
                                if (is_bonded) cycle

                                ! if not, add it (for BOTH atoms)
                                a_saved = .false.
                                b_saved = .false.
                                do l = 1, 8
                                    if (atom_bonded(i,l) == 0 .and. .not. a_saved) then
                                        atom_bonded(i,l) = j
                                        a_saved = .true.
                                    endif
                                    if (atom_bonded(j,l) == 0 .and. .not. b_saved) then
                                        atom_bonded(j,l) = i
                                        b_saved = .true.
                                    endif
                                    if (a_saved .and. b_saved) exit
                                enddo
                                bond_idx = bond_idx + 1
                                bonds(bond_idx, 1) = min(i,j)
                                bonds(bond_idx, 2) = max(i,j)
                            endif
                        enddo
                    enddo
                enddo
            enddo
        enddo
    enddo

    nbonds = bond_idx

    bonds = bonds - 1

    if (allocated(atom_cell)) deallocate(atom_cell)
    if (allocated(cell_counts)) deallocate(cell_counts)
    if (allocated(cell_atoms)) deallocate(cell_atoms)

end subroutine find_bonds_spacepartition
