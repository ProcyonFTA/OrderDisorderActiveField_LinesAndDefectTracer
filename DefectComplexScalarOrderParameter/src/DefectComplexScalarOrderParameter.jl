module DefectComplexScalarOrderParameter

    using HDF5, SpecialMatrices, WriteVTK

    function import_phi_field(path_to_h5)
        # Input : string --> path to an HDF5 file with two fields "pre" and "pim" for real and imaginary part of a field 
        # Returns : complex array --> complex field φ
        phi = h5read(path_to_h5,"pre") .+ 1im.*h5read(path_to_h5,"pim")
        return phi
    end

    ### Defects ###

    function TwoPi_to_MinusPiPi(value)
        # Input : Float --> angle in [0,2π] 
        # Returns : Float --> angle in [-π,π] 
        if value > pi
            return value-2*pi
        else
            return value 
        end
    end
    
    function dif_angle(c1,c2)
        # Input : two complex numbers c1,c2 --> values of the field on 2 consecutices vertices 
        # Returns : Float --> angle difference in [0,2π] between the arguments of c1 and c2 
        value = (angle(c1)-angle(c2)+2*pi)%(2*pi)
        return value
    end
    
    function update_defect_cube!(defect_cube,cubic_dataset,i,j,k)
        # Input : defect_cube --> complex field size (nx-1,ny-1,nz-1,3) : result hypercube for the Berg algorithm defect finder
        #         cubic_dataset --> complex field size (nx,ny,nz) : complex field φ
        #         (i,j,k) --> Int : position of a vertice
        # Update : defect_cube in all vertices starting from (i,j,k)
        p_theta_x = TwoPi_to_MinusPiPi(dif_angle(cubic_dataset[i+1,j,k],cubic_dataset[i,j,k]))
        p_theta_y = TwoPi_to_MinusPiPi(dif_angle(cubic_dataset[i,j+1,k],cubic_dataset[i,j,k]))
        p_theta_z = TwoPi_to_MinusPiPi(dif_angle(cubic_dataset[i,j,k+1],cubic_dataset[i,j,k]))
    
        m_theta_x = TwoPi_to_MinusPiPi(dif_angle(cubic_dataset[i,j,k],cubic_dataset[i+1,j,k]))
        m_theta_y = TwoPi_to_MinusPiPi(dif_angle(cubic_dataset[i,j,k],cubic_dataset[i,j+1,k]))
        m_theta_z = TwoPi_to_MinusPiPi(dif_angle(cubic_dataset[i,j,k],cubic_dataset[i,j,k+1]))   
    
        defect_cube[i,j,k,3] += (p_theta_x+m_theta_y)/(2*pi)
        defect_cube[i,j,k,2] += (p_theta_x+m_theta_z)/(2*pi)
        defect_cube[i,j,k,1] += (p_theta_y+m_theta_z)/(2*pi)
    
        if i>1
            @inbounds defect_cube[i-1,j,k,3] += p_theta_y/(2*pi)
            @inbounds defect_cube[i-1,j,k,2] += p_theta_z/(2*pi)
        end
        if j>1 
            @inbounds defect_cube[i,j-1,k,1] += p_theta_z/(2*pi)
            @inbounds defect_cube[i,j-1,k,3] += m_theta_x/(2*pi)
        end
        if k>1
            @inbounds defect_cube[i,j,k-1,2] += m_theta_x/(2*pi)
            @inbounds defect_cube[i,j,k-1,1] += m_theta_y/(2*pi)
        end
    
        nothing
    end
    
    function cpu_defect_cube!(cubic_dataset;defect_map=zeros(Float64,255,255,255,3))
        # CPU Implementation
        # Input : cubic_dataset --> complex field size (nx,ny,nz) : complex field φ
        # Returns : defect_cube --> complex field size (nx-1,ny-1,nz-1,3) : result hypercube for the Berg algorithm defect finder
        nx,ny,nz = size(cubic_dataset)
        fill!(defect_map,0.0+1im*0.0)
        for i=1:nx-1
            for j=1:ny-1
                for k=1:nz-1
                    update_defect_cube!(defect_map,cubic_dataset,i,j,k)
                end
            end
        end
    end

    function transform_ijkc_to_xyz!(new_points,cartesian_indice,i;nx=255,ny=nx,nz=nx,lx=nx,ly=ny,lz=nz)
        # the points are located at the center of a vertices, no on the edge
        if cartesian_indice[4]==1
            new_points[i,1] = cartesian_indice[1]*lx/nx
            new_points[i,2] = (cartesian_indice[2]+0.5)*ly/ny
            new_points[i,3] = (cartesian_indice[3]+0.5)*lz/nz
        elseif cartesian_indice[4]==2
            new_points[i,1] = (cartesian_indice[1]+0.5)*lx/nx
            new_points[i,2] = (cartesian_indice[2])*ly/ny
            new_points[i,3] = (cartesian_indice[3]+0.5)*lz/nz
        elseif cartesian_indice[4]==3
            new_points[i,1] = (cartesian_indice[1]+0.5)*lx/nx
            new_points[i,2] = (cartesian_indice[2]+0.5)*ly/ny
            new_points[i,3] = (cartesian_indice[3])*lz/nz
        end
    end
    
    function defect_points(defect;threshold=0.85,nx=255,ny=nx,nz=nx,lx=nx,ly=ny,lz=nz,gpu=false)
        # Input : defect_cube --> complex field size (nx-1,ny-1,nz-1,3) : result hypercube for the Berg algorithm defect finder
        # Returns : points --> matrix of size (n,3) : points where a defect line crosses
        index = findall(x -> x>threshold,abs.(defect))
        new_points = zeros(Float64,length(index),3)
        for (i,cartesian_indice) in zip(1:length(index),index)
            transform_ijkc_to_xyz!(new_points,cartesian_indice,i)
        end
        return new_points
    end

    function norm_array_indices(array,i1,i2)
        s = 0
        for i=1:3
            s+=(array[i1,i]-array[i2,i])^2
        end
        return sqrt(s)
    end
    
    function scalar_product_array_indices(array,i1,i2)
        s = 0
        for i=1:3
            s+=array[i1,i]*array[i2,i]
        end
        return s
    end
    
    function check_prolongeability(index::Int64,points::Array{Float64,2},liberty::BitVector;dl=1.0)
        # check if the points a ths position index of the list is close enough to another point in the list to be linked through a line
        # return the index of the next point on the line, 0 if it cannot be linked 
        n,d = size(points) 
        for k=1:n
            @inbounds if liberty[k]
                if (norm_array_indices(points,index,k)<=dl)
                #if (norm(points[index,:].- points[k,:])<=dl)
                    return k
                end
            end
        end
        return 0
    end
    
    function s_line_indices(indices,points;i_1=1,i_n=5,spot=1)
        # compute the cumulative array of the length along a line going through the points at the indices
        s = zeros(Float64,i_n-i_1+1)
        for j=1:i_n-i_1
            @inbounds s[j+1]=s[j]+norm_array_indices(points,indices[i_1+j],indices[i_1+j-1])
        end
        return s.-s[spot]
    end
    
    function find_coeff_non_uniform1(x)
        A = transpose(Vandermonde(x))
        b = zeros(Float64,length(x))
        b[2]=1
        return A\b
    end
    
    function non_uniform_first_derivative(x,indices,points;d=1,coeff=nothing,p=1)
        if coeff === nothing
            println("ALERT")
            coeff = find_coeff_non_uniform1(x;p=p)
        end
        result = 0
        for k=1:length(coeff)
            result+= coeff[k]* points[indices[k],d]
        end
        return result #sum(coeff.*y)
    end
    
    function direction_point(indices,points;p=1)
        result = zeros(Float64,size(points)[2])
        s = s_line_indices(indices,points;i_1=i_1,i_n=i_n,spot=p) #carabistouille
        c = find_coeff_non_uniform1(x5)
        for d=1:size(points)[1]
            result[d] = non_uniform_first_derivative(s,indices,points;d=d,coeff=c,p=p)
        end
        return result
    end
    
    function direction_at_point(indices,points;i_1=1,i_n=5,p=1)
        direction = zeros(Float64,size(points)[2])
        if length(indices)>1
            s = s_line_indices(indices,points;i_1=i_1,i_n=i_n,spot=p)
            c = find_coeff_non_uniform1(s)
            for d=1:size(points)[2]
                direction[d] = non_uniform_first_derivative(s,indices,points;d=d,coeff=c,p=p)
            end    
        end
        return direction 
    end
    
    function get_extremities_line(l,infos_line)
        #println(typeof(infos_line))
        return infos_line[l]
    end
    
    function compatibility_lines!(l1::Int64,l2::Int64,points::Array{Float64,2},extremity_direction::Array{Float64, 2},join_index::Tuple{Int,Int};dl=1.5,max_dl=2.0,index_az1=(0,0),index_az2=(0,0))
        # check if two lines l1 and l2 can be merged
        # return the extremities where the link appends, [0,0] otherwise
        if index_az1 == (0,0)
            index_az1 .= get_extremities_line(l1,infos_line)
        end    
        if index_az2 == (0,0)
            index_az2 .= get_extremities_line(l2,infos_line)
        end
        if (norm_array_indices(points,index_az1[1],index_az2[1])<=dl)||((norm_array_indices(points,index_az1[1],index_az2[1])<=max_dl)&&(-scalar_product_array_indices(extremity_direction,index_az1[1],index_az2[1])>=0))## both begining
            #println(norm(points[index_az1[1],:].-points[index_az2[1],:]))
            join_index = (1,1)
        elseif (norm_array_indices(points,index_az1[1],index_az2[2])<=dl)||((norm_array_indices(points,index_az1[1],index_az2[2])<=max_dl)&&(scalar_product_array_indices(extremity_direction,index_az1[1],index_az2[2])>=0)) ## beginning + end
            #println(norm(points[index_az1[1],:].-points[index_az2[2],:]))
            join_index = (1,2)
        elseif (norm_array_indices(points,index_az1[2],index_az2[1])<=dl)||((norm_array_indices(points,index_az1[2],index_az2[1])<=max_dl)&&(scalar_product_array_indices(extremity_direction,index_az1[2],index_az2[1])>=0)) ## end + beginning
            #println(norm(points[index_az1[2],:].-points[index_az2[1],:]))
            join_index = (2,1)
        elseif (norm_array_indices(points,index_az1[2],index_az2[2])<=dl)||((norm_array_indices(points,index_az1[2],index_az2[2])<=max_dl)&&(-scalar_product_array_indices(extremity_direction,index_az1[2],index_az2[2])>=0)) ## end + end
            #println(norm(points[index_az1[2],:].-points[index_az2[2],:]))
            join_index = (2,2)
        else
            join_index = (0,0)
        end
    end
    
    function update_merge!(join_index,l1,l2,inline,extremity_direction,infos_line;index_az1=[0,0],index_az2=[0,0])
        m = maximum(collect(keys(infos_line));init=0)+1
        if join_index == (1,1)
            infos_line[m]=(index_az2[2],index_az1[2])
            a = inline[index_az2[2],2]
            for d=1:3
                extremity_direction[index_az2[2],d] = -extremity_direction[index_az2[2],d]
            end
            for k=1:size(inline)[1]
                if inline[k,1]==l1
                    inline[k,2] += a
                    inline[k,1] = m
                elseif inline[k,1]==l2
                    inline[k,2] = (a+1) - inline[k,2]
                    inline[k,1] = m
                end
            end
        elseif join_index == (1,2)
            infos_line[m]=(index_az2[1],index_az1[2])
            a = inline[index_az2[2],2]
            for k=1:size(inline)[1]
                if inline[k,1]==l1
                    inline[k,2] += a
                    inline[k,1] = m
                elseif inline[k,1]==l2
                    inline[k,1] = m
                end
            end
        elseif join_index == (2,1)
            infos_line[m]=(index_az1[1],index_az2[2])
            a = inline[index_az1[2],2]
            for k=1:size(inline)[1]
                if inline[k,1]==l2
                    inline[k,2] += a
                    inline[k,1] = m
                elseif inline[k,1]==l1
                    inline[k,1] = m
                end
            end
        elseif join_index == (2,2)
            infos_line[m]=(index_az1[1],index_az2[1])
            a,b = inline[index_az2[2],2],inline[index_az1[2],2]
            for d=1:3
                extremity_direction[index_az2[1],d] = -extremity_direction[index_az2[1],d]
            end
            for k=1:size(inline)[1]
                if inline[k,1]==l2
                    inline[k,2] = (a+1+b) - inline[k,2]
                    inline[k,1] = m
                elseif inline[k,1]==l1
                    inline[k,1] = m
                end
            end
        end
        delete!(infos_line,l1)
        delete!(infos_line,l2)
    end
    
    function merge_lines!(l1::Int64,l2::Int64,inline::Array{Int64, 2},extremity_direction::Array{Float64, 2},infos_line,join_index::Tuple{Int,Int};max_dl=2.0,index_az1=[0,0],index_az2=[0,0])
        # merge the lines l1 and l2 and, creates a new entry in the dictionnary infos_line and delete the two old ones
        if join_index != (0,0)
            update_merge!(join_index,l1,l2,inline,extremity_direction,infos_line;index_az1=index_az1,index_az2=index_az2)
        end
    end
    
    function prolonge_lines!(l::Int64,points::Array{Float64,2},inline::Array{Int64, 2},extremity_direction::Array{Float64, 2},infos_line,join_index::Tuple{Int,Int};max_dl=2.0)
        # after a line has been drawn, check if it can be prolongated to another existing line ; if its the case merge both line and iterates once again with the new merged line
        list_id_lines = collect(keys(infos_line))
        list_id_lines = list_id_lines[list_id_lines.>0]
        k = 1
        while k<=length(list_id_lines)
            #println(list_id_lines)
            if l != list_id_lines[k]
                #println(list_id_lines[k])
                #println(l)
                index_az1 = get_extremities_line(l,infos_line)
                index_az2 = get_extremities_line(list_id_lines[k],infos_line)
                join_index = compatibility_lines!(l,list_id_lines[k],points,extremity_direction,join_index;max_dl=max_dl,index_az1=index_az1,index_az2=index_az2)
                #println(join_index)
                if join_index != (0,0)
                    merge_lines!(l,list_id_lines[k],inline,extremity_direction,infos_line,join_index;max_dl=max_dl,index_az1=index_az1,index_az2=index_az2)
                    k=0
                    list_id_lines = collect(keys(infos_line))
                    list_id_lines = list_id_lines[list_id_lines.>0]
                    l = maximum(list_id_lines)
                end
            end
            k+=1
        end
    end
    
    function draw_line!(initial_index::Int64,points::Array{Float64,2},inline::Array{Int64, 2},liberty::BitVector,extremity_direction::Array{Float64, 2},infos_line,join_index::Tuple{Int,Int},indices::Array{Int64, 1};dl=1.0,max_dl=2.0)
        # draw a line in the inline matrix + add the extremity vectors and update the liberty vector
        m = maximum(collect(keys(infos_line));init=0)+1
        n,d = size(points)
        liberty[initial_index]=false
        p = 1
        @views inline[initial_index,:]=[m,p]
        indices[1] = initial_index
        k = check_prolongeability(initial_index,points,liberty;dl=dl)
        while k>0
            p+=1
            inline[k,:]=[m,p]
            liberty[k]=false
            indices[p] = k 
            k = check_prolongeability(k,points,liberty;dl=dl)
        end
        infos_line[m]=(indices[1],indices[p])
        @views extremity_direction[indices[1],:]=direction_at_point(indices[1:p],points;i_1=1,i_n=minimum([5,p]),p=1)
        q = maximum([p-4,1])
        @views extremity_direction[indices[p],:]=direction_at_point(indices[1:p],points;i_1=q,i_n=p,p=p-q+1)
        prolonge_lines!(m,points,inline,extremity_direction,infos_line,join_index;max_dl=max_dl)
        #println(collect(keys(infos_line)))
    end
    
    function line_collection(defect_points::Array{Float64,2};dl=1.0,max_dl=2.0,progress_bar=false)
        # Input : defect_points --> array of floats size (n,3) : defect points obtain with the package Defects
        # Output : inline --> array of Int (n,2) : the first component [i,1] corresponds to the index of the line to which the point [i,:] belongs
        #          infos_line --> dict with n entries : keys are the line id, then infos_line[key] = [index of the first point of the line, index of the last point of the line]
        #          extremity direction --> array Float (n,3) : if computed, extremity_direction[i,:] gives the vector tangent to the line at point [i,:]     
        n,d = size(defect_points)
        if progress_bar==true
            p = Progress(n; desc="Drawing lines...", dt=30.0)
        end
        inline = zeros(Int64,n,2)
        liberty = trues(n)
        join_index=(0,0)
        m=1
        infos_line = Dict(0=>(0,0))
        indices = zeros(Int64,n)
        #println(typeof(infos_line))
        extremity_direction = zeros(Float64,n,3)
        for k=1:n
            if progress_bar==true
                next!(p)
            end
            #println(k)
            @inbounds if liberty[k]
                #lk = maximum(collect(keys(infos_line)))+1
                #println(lk)
                draw_line!(k,defect_points,inline,liberty,extremity_direction,infos_line,join_index,indices;dl=dl,max_dl=max_dl)
            end
        end
        return inline, infos_line, extremity_direction
    end

    function get_sorted_points_line(points,inline,id_line)
        sp = points[inline[:,1].==id_line,:]
        si = inline[inline[:,1].==id_line,:]
        order = sortperm(si[:,2])
        return sp[order,:],si[order,:]
    end

    function cumulative_length(points,inline,id_line;sorted_points=false)
        # compute the cumulative array of the length along a line going through the points at the indices
        p,i1 = get_sorted_points_line(points,inline,id_line)
        s = zeros(Float64,size(p)[1])
        if size(p)[1]==1
            if sorted_points
                return [1.0],p,i1
            else
                return [1.0]
            end
        else
            for j=2:size(p)[1]
                s[j]=s[j-1]+norm_array_indices(p,j,j-1)
            end
            if sorted_points
                return s,p,i1
            else
                return s
            end
        end
    end

    function scaffold_points_lines(points,inline)
        id_lines = union(inline[:,1])
        offset = 0
        x = Float64[]
        y = Float64[]
        z = Float64[]
        ids = Float64[]
        icr = Float64[]
        lengths = Float64[]
        w = Array{String,1}(undef,length(id_lines))
        for (k,id) in enumerate(id_lines)
            s,sp,si = cumulative_length(points,inline,id;sorted_points=true)
            append!(x,sp[:,1])
            append!(y,sp[:,2])
            append!(z,sp[:,3])
            n_points = length(s)
            append!(ids,id.*ones(Int64,n_points))
            cl = hypot(sp[1,1]-sp[end,1],sp[1,2]-sp[end,2],sp[1,3]-sp[end,3])/s[end]
            append!(icr,cl.*ones(Int64,n_points))
            append!(lengths,s[end].*ones(Int64,n_points))            
            w[k] = "$n_points $(join(offset .+ (0:n_points-1), " "))"
            offset += n_points
        end
        return (x,y,z),w,(ids,icr,lengths)
    end

    function write_lines_vtk(points,inline,path)
        (x,y,z),w,(ids,icr,lengths) = scaffold_points_lines(points,inline)
        id_lines = union(inline[:,1])
        nbr_pts = length(x)
        nbr_lns = length(id_lines)
        open(path,"w") do f
            println(f, "# vtk DataFile Version 2.0")
            println(f, "Multiple Lines Data")
            println(f, "ASCII")
            println(f, "DATASET POLYDATA")
            println(f, "POINTS $nbr_pts float")
            for k=1:length(x)
                println(f, "$(x[k]) $(y[k]) $(z[k])")
            end

            println(f, "LINES $nbr_lns $(nbr_lns+nbr_pts)")
            for k=1:length(id_lines)
                println(f, w[k])
            end

            println(f,"POINT_DATA $nbr_pts")

            println(f,"SCALARS id float")
            println(f,"LOOKUP_TABLE default")
            for k=1:length(x)
                scalar_value =  ids[k]
                println(f,"$scalar_value")
            end
            
            println(f,"SCALARS icr float")
            println(f,"LOOKUP_TABLE default")
            for k=1:length(x)
                scalar_value =  icr[k]
                println(f,"$scalar_value")
            end

            println(f,"SCALARS length float")
            println(f,"LOOKUP_TABLE default")
            for k=1:length(x)
                scalar_value =  lengths[k]
                println(f,"$scalar_value")
            end
        end
        return nothing
    end

end # module DefectComplexScalarOrderParameter
